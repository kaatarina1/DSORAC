export class DepthMap {
	constructor(canvas, device, depthTexture) {
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
		this.depthTexture = depthTexture;

		this.depthStorageBuffer = this.device.createBuffer({
			size: this.width * this.height * 4, // Float32 per pixel
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.readbackBuffer = this.device.createBuffer({
			size: this.width * this.height * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		this.computeShader = device.createShaderModule({
			code: `
              @group(0) @binding(0) var depthTexture: texture_depth_2d;
              @group(0) @binding(1) var<storage, read_write> outputDepths: array<f32>;
              
              @compute @workgroup_size(16, 16)
              fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let dimensions = textureDimensions(depthTexture);
                if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
                  return;
                }
                
                let texCoord = vec2i(global_id.xy);
                let depth = textureLoad(depthTexture, texCoord, 0);
                
                let index = global_id.y * dimensions.x + global_id.x;
                outputDepths[index] = depth;
              }
            `,
		});

		this.computePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.computeShader,
				entryPoint: "main",
			},
		});

		this.computeBindGroup = this.device.createBindGroup({
			layout: this.computePipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: this.depthTexture.createView({
						format: "depth32float", // Explicit format
						aspect: "depth-only", // ← Important for depth textures
					}),
				},
				{
					binding: 1,
					resource: {
						buffer: this.depthStorageBuffer,
					},
				},
			],
		});
	}

	async extractDepthValues() {
		// Create a new command encoder or use your existing one
		const commandEncoder = this.device.createCommandEncoder();

		// After the depth texture is created, run the compute pass to extract depth values
		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.computePipeline);
		computePass.setBindGroup(0, this.computeBindGroup);

		// Calculate workgroup counts (ceil(dimension / workgroup_size))
		const workgroupCountX = Math.ceil(this.width / 16);
		const workgroupCountY = Math.ceil(this.height / 16);

		computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
		computePass.end();

		// Copy the depth values to the readback buffer
		commandEncoder.copyBufferToBuffer(
			this.depthStorageBuffer,
			0,
			this.readbackBuffer,
			0,
			this.width * this.height * 4
		);

		// Submit the commands
		this.device.queue.submit([commandEncoder.finish()]);

		// Wait for GPU to complete the work
		await this.device.queue.onSubmittedWorkDone();

		// Map the buffer for reading
		await this.readbackBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = this.readbackBuffer.getMappedRange();

		// Create a copy of the data (needed because buffer will be unmapped)
		const depthValues = new Float32Array(arrayBuffer.slice(0));

		// Clean up
		this.readbackBuffer.unmap();

		return depthValues;
	}

	async groupDepthIntoBins() {
		let depthValues = await this.extractDepthValues();
		depthValues.sort((a, b) => a - b); // Sort depths in ascending order

		let bins = [];
		let currentBin = [depthValues[0]];
		let threshold = 0.1; // Adjust this value for merging nearby depths

		for (let i = 1; i < depthValues.length; i++) {
			if (depthValues[i] - currentBin[0] < threshold) {
				currentBin.push(depthValues[i]);
			} else {
				bins.push([currentBin[0], currentBin[currentBin.length - 1]]);
				currentBin = [depthValues[i]];
			}
		}
		if (currentBin.length > 0) {
			bins.push([currentBin[0], currentBin[currentBin.length - 1]]);
		}

		console.log("Adaptive depth bins:", bins);
		return bins;
	}
}
