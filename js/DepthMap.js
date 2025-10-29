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
	// async groupDepthIntoBins() {
	// 	let depthValues = await this.extractDepthValues();

	// 	const near = 0.1;
	// 	const far = 2000.0;

	// 	// Convert nonlinear depth to linear distance
	// 	let linearDepths = depthValues
	// 		.filter(d => d > 0.001 && d < 0.999)
	// 		.map(d => (2.0 * near * far) / (far + near - (2.0 * d - 1.0) * (far - near)));

  
	// 	// Filter out invalid depth values
	// 	depthValues = depthValues.filter(d => d > 0.001 && d < 0.999);
		
	// 	let minDepth = Infinity;
	// 	let maxDepth = -Infinity;
	// 	for (let i = 0; i < depthValues.length; i++) {
	// 	  if (depthValues[i] < minDepth) minDepth = depthValues[i];
	// 	  if (depthValues[i] > maxDepth) maxDepth = depthValues[i];
	// 	}
		
	// 	const histogramBins = 100;
	// 	const binWidth = (maxDepth - minDepth) / histogramBins;
		
	// 	// Count frequencies
	// 	const histogram = new Array(histogramBins).fill(0);
	// 	for (const depth of depthValues) {
	// 	  const binIndex = Math.min(
	// 		Math.floor((depth - minDepth) / binWidth),
	// 		histogramBins - 1
	// 	  );
	// 	  histogram[binIndex]++;
	// 	}
		
	// 	// Find peaks in the histogram (areas of high point density)
	// 	const peaks = [];
	// 	let maxFrequency = 0;
	// 	for (let i = 0; i < histogram.length; i++) {
	// 	  if (histogram[i] > maxFrequency) maxFrequency = histogram[i];
	// 	}
		
	// 	const peakThreshold = maxFrequency * 0.1;
		
	// 	for (let i = 1; i < histogramBins - 1; i++) {
	// 	  if (histogram[i] > peakThreshold && 
	// 		  histogram[i] > histogram[i - 1] && 
	// 		  histogram[i] > histogram[i + 1]) {
	// 		const peakDepth = minDepth + (i + 0.5) * binWidth;
	// 		peaks.push(peakDepth);
	// 	  }
	// 	}
		
	// 	// If no significant peaks found, fall back to uniform bins
	// 	if (peaks.length < 4) {
	// 	  const numBins = 15; 
	// 	  const uniformBins = [];
	// 	  for (let i = 0; i < numBins; i++) {
	// 		const start = minDepth + (maxDepth - minDepth) * (i / numBins);
	// 		const end = minDepth + (maxDepth - minDepth) * ((i + 1) / numBins);
	// 		uniformBins.push([start, end]);
	// 	  }
	// 	  return uniformBins;
	// 	}
		
	// 	// Sort peaks by depth
	// 	peaks.sort((a, b) => a - b);
		
	// 	const bins = [];

	// 	// const firstPeakLowerBound = Math.max(0, (peaks[0] + (peaks[1] || maxDepth)) / 2);
	// 	// if (firstPeakLowerBound > 1e-2) {
	// 	// 	bins.push([0, firstPeakLowerBound]);
	// 	// }

	// 	for (let i = 0; i < peaks.length; i++) {
	// 	  const peakDepth = peaks[i];
		  
	// 	  let binStart, binEnd;
		  
	// 	  if (i === 0) {
	// 		binStart = minDepth;
	// 		const nextPeak = peaks[i + 1] || maxDepth;
	// 		binEnd = (peakDepth + nextPeak) / 2;
	// 	  } else if (i === peaks.length - 1) {
	// 		const prevPeak = peaks[i - 1];
	// 		binStart = (prevPeak + peakDepth) / 2;
	// 		binEnd = maxDepth;
	// 	  } else {
	// 		const prevPeak = peaks[i - 1];
	// 		const nextPeak = peaks[i + 1];
	// 		binStart = (prevPeak + peakDepth) / 2;
	// 		binEnd = (peakDepth + nextPeak) / 2;
	// 	  }
		  
	// 	  bins.push([binStart, binEnd]);
	// 	}
		
	// 	console.log("Adaptive depth clusters:", bins);
	// 	return bins;
	// }

	async groupDepthIntoBins({
  histogramBins = 128,
  peakThresholdFactor = 0.01,
  smoothKernelSize = 3,
  near = 0.1,
  far = 2000.0,
  nearBias = 0.7 // <1 -> denser in front; 1 = no bias; >1 -> denser at far
} = {}) {
  // read depth buffer (assumes extractDepthValues returns array of depth in [0,1])
  let depthValues = await this.extractDepthValues();
  // filter out invalid values
  let valid = depthValues.filter(d => Number.isFinite(d) && d > 0.0005 && d < 0.9995);
  if (valid.length === 0) {
    console.warn("no valid depth samples");
    return [[0, 1]]; // fallback single bin (depth buffer domain)
  }

  // --- 1) linearize depth values (convert nonlinear depth -> view-space distance)
  // convert depth in [0,1] to NDC z in [-1,1]:
  // z_ndc = depth * 2 - 1
  // view_z = (2*near*far) / (far + near - z_ndc*(far - near))
  const linearize = (d) => {
    const z_ndc = d * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z_ndc * (far - near));
  };
  const linearDepths = valid.map(d => linearize(d));

  function linearToDepthBuffer(zView) {
    // invert the linearize steps:
    // z_ndc = (far + near - 2*near*far / zView) / (far - near)
    const term = (2.0 * near * far) / zView;
    const z_ndc = (far + near - term) / (far - near);
    const depth = (z_ndc + 1.0) * 0.5; // back to [0,1]
    return depth;
  }

  // --- 2) optional near-bias mapping to make bins denser near camera
  // We remap normalized linear depth by raising to power alpha in (0,1) to expand small distances.
  // lower nearBias -> stronger bias towards front (e.g. 0.5), 1 = no bias.
    // normalize to 0..1 then pow, then denormalize
	let minL = Infinity;
	let maxL = -Infinity;
	for (let i = 0; i < linearDepths.length; i++) {
		if (linearDepths[i] < minL) minL = linearDepths[i];
		if (linearDepths[i] > maxL) maxL = linearDepths[i];
	}
    // const minL = Math.min(...linearDepths);
    // const maxL = Math.max(...linearDepths);
const range = Math.max(1e-6, maxL - minL);
    const biasStrength = 5.0; // 1 = uniform, >1 = denser near

	const remapped = linearDepths.map(v => {
	// log scale optional, comment out if not desired
	const logMin = Math.log(minL + 1e-6);
	const logMax = Math.log(maxL + 1e-6);
	const logRange = logMax - logMin;
	const norm = (Math.log(v + 1e-6) - logMin) / logRange; // 0..1
	const biased = Math.pow(norm, 1 / biasStrength);
	return biased * range + minL;
	});

  // --- 3) build histogram on remapped (linear) distances
	let minDepth = Infinity;
	let maxDepth = -Infinity;
	for (let i = 0; i < remapped.length; i++) {
	  if (remapped[i] < minDepth) minDepth = remapped[i];
	  if (remapped[i] > maxDepth) maxDepth = remapped[i];
	}
	const binWidth = (maxDepth - minDepth) / histogramBins;
	const histogram = new Array(histogramBins).fill(0);
	for (const d of remapped) {
	const idx = Math.min(
		Math.floor((d - minDepth) / (binWidth || 1e-6)),
		histogramBins - 1
	);
	histogram[idx]++;
	}
  // --- 4) smooth histogram to reduce noise (simple box blur)
  if (smoothKernelSize > 1) {
    const half = Math.floor(smoothKernelSize / 2);
    const smoothed = new Array(histogramBins).fill(0);
    for (let i = 0; i < histogramBins; i++) {
      let sum = 0, cnt = 0;
      for (let k = -half; k <= half; k++) {
        const j = i + k;
        if (j >= 0 && j < histogramBins) { sum += histogram[j]; cnt++; }
      }
      smoothed[i] = sum / Math.max(1, cnt);
    }
    for (let i = 0; i < histogramBins; i++) histogram[i] = smoothed[i];
  }

  // --- 5) find peaks
  const peaks = [];
  let maxFreq = Math.max(...histogram);
  const peakThreshold = Math.max(1, maxFreq * peakThresholdFactor);
  for (let i = 1; i < histogramBins - 1; i++) {
    if (histogram[i] > peakThreshold && histogram[i] > histogram[i - 1] && histogram[i] >= histogram[i + 1]) {
      // compute peak center in linear units
      const peakDepth = minDepth + (i + 0.5) * binWidth;
      peaks.push(peakDepth);
    }
  }

  // If not enough peaks, fall back to uniform bins (but computed in linear distance)
  if (peaks.length < 3) {
    const numBins = 10;
    const uniform = [];
    for (let i = 0; i < numBins; i++) {
      const s = minDepth + (maxDepth - minDepth) * (i / numBins);
      const e = minDepth + (maxDepth - minDepth) * ((i + 1) / numBins);
      uniform.push([s, e]);
    }
    console.log("uniform linear bins (fallback)", uniform);
	// const logBins = uniform.reverse().map(([s, e]) => [ linearToDepthBuffer(s), linearToDepthBuffer(e) ]);
	  if (uniform.length > 0) {
	const firstStart = uniform[0][0];
	if (firstStart > 0.01) {
		// Add an extra bin from 0 to the start of the first one
		uniform.unshift([0, firstStart]);
	}
	}

	return uniform;
  }

  // --- 6) turn peaks into contiguous bins (in linear distance)
  peaks.sort((a, b) => a - b);
  const binsLinear = [];
  for (let i = 0; i < peaks.length; i++) {
    let start, end;
    if (i === 0) {
      start = minDepth;
      end = (peaks[i] + (peaks[i + 1] ?? maxDepth)) / 2;
    } else if (i === peaks.length - 1) {
      start = (peaks[i - 1] + peaks[i]) / 2;
      end = maxDepth;
    } else {
      start = (peaks[i - 1] + peaks[i]) / 2;
      end = (peaks[i] + peaks[i + 1]) / 2;
    }
    binsLinear.push([start, end]);
  }

  console.log("Adaptive linear-depth clusters (linear units):", binsLinear);

    
//   const binsDepthBuffer = binsLinear.map(([s, e]) => [ linearToDepthBuffer(s), linearToDepthBuffer(e) ]);
//   if (binsDepthBuffer.length > 0) {
// 	const firstStart = binsDepthBuffer[0][0];
// 	if (firstStart > 0.01) {
// 		// Add an extra bin from 0 to the start of the first one
// 		binsDepthBuffer.unshift([0, firstStart]);
// 	}
// 	}
//   console.log("Adaptive log-depth clusters (linear units):", binsDepthBuffer);
//   return binsDepthBuffer;
  if (binsLinear.length > 0) {
	const firstStart = binsLinear[0][0];
	if (firstStart > 0.01) {
		// Add an extra bin from 0 to the start of the first one
		binsLinear.unshift([0, firstStart]);
	}
	}
  return binsLinear;
}


}
