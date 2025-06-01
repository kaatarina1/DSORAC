export class DepthMap {
	constructor(canvas, device, depthTexture) {
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
		this.depthTexture = depthTexture;

		this.depthStorageBuffer = this.device.createBuffer({
			size: this.width * this.height * 4,
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
						format: "depth32float", 
						aspect: "depth-only", 
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

	// Function for extracting depth values
	// Returns depth bins
	async extractDepthValues() {
		const commandEncoder = this.device.createCommandEncoder();

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.computePipeline);
		computePass.setBindGroup(0, this.computeBindGroup);

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

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		// Map the buffer for reading
		await this.readbackBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = this.readbackBuffer.getMappedRange();
		const depthValues = new Float32Array(arrayBuffer.slice(0));

		this.readbackBuffer.unmap();

		return depthValues;
	}

	// Creating bins based on the dapth and points density
	async groupDepthIntoBins() {
		let depthValues = await this.extractDepthValues();
  
		// Filter out invalid depth values
		depthValues = depthValues.filter(d => d > 0.001 && d < 0.999);
		
		let minDepth = Infinity;
		let maxDepth = -Infinity;
		for (let i = 0; i < depthValues.length; i++) {
		  if (depthValues[i] < minDepth) minDepth = depthValues[i];
		  if (depthValues[i] > maxDepth) maxDepth = depthValues[i];
		}
		
		const histogramBins = 100;
		const binWidth = (maxDepth - minDepth) / histogramBins;
		
		// Count frequencies
		const histogram = new Array(histogramBins).fill(0);
		for (const depth of depthValues) {
		  const binIndex = Math.min(
			Math.floor((depth - minDepth) / binWidth),
			histogramBins - 1
		  );
		  histogram[binIndex]++;
		}
		
		// Find peaks in the histogram (areas of high point density)
		const peaks = [];
		let maxFrequency = 0;
		for (let i = 0; i < histogram.length; i++) {
		  if (histogram[i] > maxFrequency) maxFrequency = histogram[i];
		}
		
		const peakThreshold = maxFrequency * 0.01;
		
		for (let i = 1; i < histogramBins - 1; i++) {
		  if (histogram[i] > peakThreshold && 
			  histogram[i] > histogram[i - 1] && 
			  histogram[i] > histogram[i + 1]) {
			const peakDepth = minDepth + (i + 0.5) * binWidth;
			peaks.push(peakDepth);
		  }
		}
		
		// If no significant peaks found, fall back to uniform bins
		if (peaks.length < 4) {
		  const numBins = 15; 
		  const uniformBins = [];
		  for (let i = 0; i < numBins; i++) {
			const start = minDepth + (maxDepth - minDepth) * (i / numBins);
			const end = minDepth + (maxDepth - minDepth) * ((i + 1) / numBins);
			uniformBins.push([start, end]);
		  }
		  return uniformBins;
		}
		
		// Sort peaks by depth
		peaks.sort((a, b) => a - b);
		
		const bins = [];

		const firstPeakLowerBound = Math.max(0, (peaks[0] + (peaks[1] || maxDepth)) / 2);
		if (firstPeakLowerBound > 1e-2) {
			bins.push([0, firstPeakLowerBound]);
		}

		for (let i = 0; i < peaks.length; i++) {
		  const peakDepth = peaks[i];
		  
		  let binStart, binEnd;
		  
		  if (i === 0) {
			binStart = minDepth;
			const nextPeak = peaks[i + 1] || maxDepth;
			binEnd = (peakDepth + nextPeak) / 2;
		  } else if (i === peaks.length - 1) {
			const prevPeak = peaks[i - 1];
			binStart = (prevPeak + peakDepth) / 2;
			binEnd = maxDepth;
		  } else {
			const prevPeak = peaks[i - 1];
			const nextPeak = peaks[i + 1];
			binStart = (prevPeak + peakDepth) / 2;
			binEnd = (peakDepth + nextPeak) / 2;
		  }
		  
		  bins.push([binStart, binEnd]);
		}
		
		console.log("Adaptive depth clusters:", bins);
		return bins;
	}
}
