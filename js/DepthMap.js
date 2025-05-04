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
		// Get the raw depth values
		let depthValues = await this.extractDepthValues();
  
		// Filter out invalid depth values (0 or 1 typically represent invalid measurements)
		depthValues = depthValues.filter(d => d > 0.001 && d < 0.999);
		
		// Find min/max without using spread operator
		let minDepth = Infinity;
		let maxDepth = -Infinity;
		for (let i = 0; i < depthValues.length; i++) {
		  if (depthValues[i] < minDepth) minDepth = depthValues[i];
		  if (depthValues[i] > maxDepth) maxDepth = depthValues[i];
		}
		
		// Create a histogram to analyze depth distribution
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
		// Calculate the maximum value in histogram without using Math.max(...) 
		let maxFrequency = 0;
		for (let i = 0; i < histogram.length; i++) {
		  if (histogram[i] > maxFrequency) maxFrequency = histogram[i];
		}
		
		const peakThreshold = maxFrequency * 0.008; // Adjust sensitivity
		
		for (let i = 1; i < histogramBins - 1; i++) {
		  if (histogram[i] > peakThreshold && 
			  histogram[i] > histogram[i-1] && 
			  histogram[i] > histogram[i+1]) {
			// Found a peak
			const peakDepth = minDepth + (i + 0.5) * binWidth;
			peaks.push(peakDepth);
		  }
		}
		
		// If no significant peaks found, fall back to uniform bins
		if (peaks.length < 2) {
		  console.log("No significant depth layers detected, using uniform bins");
		  const numBins = 5; // Fallback to 5 uniform bins
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
		
		// Create bins around peaks with adaptive widths
		const bins = [];

		// Add initial bin from 0 to first peak's lower bound
		const firstPeakLowerBound = Math.max(0, (peaks[0] + (peaks[1] || maxDepth)) / 2);
		bins.push([0, firstPeakLowerBound]);

		for (let i = 0; i < peaks.length; i++) {
		  const peakDepth = peaks[i];
		  
		  // Calculate adaptive width based on distance to neighboring peaks
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

		// Add final bin from last peak's upper bound to 1
		const lastPeakUpperBound = Math.min(1, (peaks[peaks.length - 1] + (peaks[peaks.length - 2] || minDepth)) / 2);
		bins.push([lastPeakUpperBound, 1]);
		
		console.log("Adaptive depth clusters:", bins);
		return bins;
	}
}
