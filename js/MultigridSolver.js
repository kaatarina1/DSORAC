import { Solver } from "./Solvers";
import { convertTexture } from "./Utils";

export class MultigridSolver {
	constructor(canvas, device, levels = 10, nSmooth = 20, nSolve = 10) {
		this.canvas = canvas;
		this.device = device;
		this.levels = levels;
		this.height = canvas.height;
		this.width = canvas.width;
		this.nSmooth = nSmooth;
		this.nSolve = nSolve;
		this.grid = [];
		this.restrictionPipeline = null;
		this.correctionPipeline = null;
		this.residualPipeline = null;
		this.sorSolver = new Solver(canvas, device);
		this.format = "rgba32float";
		this.duration = 0;
	}

	async initializeGrids() {
		// Create textures for each level
		for (let i = 0; i < this.levels; i++) {
			const levelWidth = Math.max(1, this.width >> i);
			const levelHeight = Math.max(1, this.height >> i);
			// console.log(`Level ${i}: ${levelWidth}x${levelHeight}`);
			this.grid[i] = {
				width: levelWidth,
				height: levelHeight,
				reconstructionRead: this.device.createTexture({
					size: [levelWidth, levelHeight],
					format: this.format,
					usage:
						GPUTextureUsage.TEXTURE_BINDING |
						GPUTextureUsage.STORAGE_BINDING |
						GPUTextureUsage.COPY_SRC |
						GPUTextureUsage.COPY_DST,
				}),
				reconstructionWrite: this.device.createTexture({
					size: [levelWidth, levelHeight],
					format: this.format,
					usage:
						GPUTextureUsage.TEXTURE_BINDING |
						GPUTextureUsage.STORAGE_BINDING |
						GPUTextureUsage.COPY_SRC |
						GPUTextureUsage.COPY_DST,
				}),
				points: this.device.createTexture({
					size: [levelWidth, levelHeight],
					format: this.format,
					usage:
						GPUTextureUsage.TEXTURE_BINDING |
						GPUTextureUsage.STORAGE_BINDING |
						GPUTextureUsage.COPY_SRC |
						GPUTextureUsage.COPY_DST,
				}),
				f: this.device.createTexture({
					size: [levelWidth, levelHeight],
					format: this.format,
					usage:
						GPUTextureUsage.TEXTURE_BINDING |
						GPUTextureUsage.STORAGE_BINDING |
						GPUTextureUsage.COPY_SRC |
						GPUTextureUsage.COPY_DST,
				}),
				temp: this.device.createTexture({
					size: [levelWidth, levelHeight],
					format: this.format,
					usage:
						GPUTextureUsage.TEXTURE_BINDING |
						GPUTextureUsage.STORAGE_BINDING |
						GPUTextureUsage.COPY_SRC |
						GPUTextureUsage.COPY_DST,
				}),
			};
		}
	}

	async createRestrictionPipeline() {
		const recrictionCode = await fetch("shaders/restriction.wgsl").then(
			(res) => res.text()
		);
		const restrictionModule = this.device.createShaderModule({
			code: recrictionCode,
		});
		return this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: restrictionModule,
				entryPoint: "main",
			},
		});
	}

	async createCorrectionPipeline() {
		const correctionCode = await fetch("shaders/correction.wgsl").then(
			(res) => res.text()
		);
		const correctionModule = this.device.createShaderModule({
			code: correctionCode,
		});
		return this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: correctionModule,
				entryPoint: "main",
			},
		});
	}

	async createResidualPipeline() {
		const residualCode = await fetch("shaders/residual.wgsl").then((res) =>
			res.text()
		);
		const residualModule = this.device.createShaderModule({
			code: residualCode,
		});
		return this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: residualModule,
				entryPoint: "main",
			},
		});
	}

	async initialize() {
		await this.initializeGrids();
		this.restrictionPipeline = await this.createRestrictionPipeline();
		this.correctionPipeline = await this.createCorrectionPipeline();
		this.residualPipeline = await this.createResidualPipeline();
	}

	async multigridSolve(captureTexture) {
		await this.initialize();

		await convertTexture(
			this.device,
			this.width,
			this.height,
			captureTexture,
			this.grid[0].points
		);

		await this.vCycle(0);

		const outputBuffer = this.device.createBuffer({
			size: this.width * this.height * 16, // 4 bytes per pixel (RGBA)
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const commandEncoder = this.device.createCommandEncoder();

		commandEncoder.copyTextureToBuffer(
			{
				texture: this.grid[0].reconstructionRead,
				mipLevel: 0,
				origin: { x: 0, y: 0, z: 0 },
			},
			{
				buffer: outputBuffer,
				bytesPerRow: this.width * 16,
				rowsPerImage: this.height,
			},
			[this.width, this.height, 1]
		);

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		await outputBuffer.mapAsync(GPUMapMode.READ);
		const outputArrayBuffer = outputBuffer.getMappedRange();
		const floatData = new Float32Array(outputArrayBuffer.slice(0));

		// Convert to Uint8Array for display (optional)
		const outputImage = new Uint8Array(this.width * this.height * 4);
		for (let i = 0; i < floatData.length; i += 4) {
			// Simple normalization for display
			outputImage[i] = Math.max(0, Math.min(255, floatData[i] * 255));
			outputImage[i + 1] = Math.max(
				0,
				Math.min(255, floatData[i + 1] * 255)
			);
			outputImage[i + 2] = Math.max(0, Math.min(255, floatData[i + 2] * 255));
			outputImage[i + 3] = Math.max(
				0,
				Math.min(255, floatData[i + 3] * 255)
			);
		}

		console.log('Multigrid time: ', this.duration);

		outputBuffer.unmap();
		this.destroy();

		return outputImage;
	}

	async vCycle(level) {
		const {
			points,
			temp,
		} = this.grid[level];

		this.duration += await this.smooth(level, this.nSmooth);

		this.duration += await this.computeResidual(level);

		this.duration += await this.restrict(level, temp, this.grid[level + 1].f, false);
		this.duration += await this.restrict(level, points, this.grid[level + 1].points, true);

		if (level + 2 < this.levels) {
			await this.vCycle(level + 1);
		} else {
			this.duration += await this.smooth(level + 1, this.nSolve);
		}

		this.duration += await this.correct(level);
		[
			this.grid[level].reconstructionRead,
			this.grid[level].reconstructionWrite,
		] = [
			this.grid[level].reconstructionWrite,
			this.grid[level].reconstructionRead,
		];

		this.duration += await this.smooth(level, this.nSmooth);
	}

	async smooth(level, iterations) {
		const {
			reconstructionRead,
			reconstructionWrite,
			points,
			f,
			width,
			height,
		} = this.grid[level];

		// Use SOR solver for smoothing
		this.sorSolver.maxIterations = iterations;
		this.sorSolver.width = width;
		this.sorSolver.height = height;
		let duration = await this.sorSolver.sorRedBlack(
			points,
			reconstructionRead,
			reconstructionWrite,
			f
		);

		return duration;
	}

	async computeResidual(level) {
		const { reconstructionRead, points, temp, f, width, height } =
			this.grid[level];
		
		const textureSampler = this.device.createSampler({
			magFilter: 'linear',  
			minFilter: 'linear',  
			addressModeU: 'clamp-to-edge', 
			addressModeV: 'clamp-to-edge',
		});

		const capacity = 3;//Max number of timestamps we can store
		const querySet = this.device.createQuerySet({
			type: "timestamp",
			count: capacity,
		});
		const queryBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage: GPUBufferUsage.QUERY_RESOLVE 
			| GPUBufferUsage.STORAGE
			| GPUBufferUsage.COPY_SRC
			| GPUBufferUsage.COPY_DST,
		});
		const resultBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const commandEncoder = this.device.createCommandEncoder();
		const bindingGroup = this.device.createBindGroup({
			layout: this.residualPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: points.createView(),
				},
				{
					binding: 1,
					resource: reconstructionRead.createView(),
				},
				{
					binding: 2,
					resource: f.createView(),
				},
				{
					binding: 3,
					resource: textureSampler,
				},
				{
					binding: 4,
					resource: temp.createView(),
				},
			],
		});

		commandEncoder.writeTimestamp(querySet, 0);
		const pass = commandEncoder.beginComputePass();
		pass.setPipeline(this.residualPipeline);
		pass.setBindGroup(0, bindingGroup);
		pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
		pass.end();
		commandEncoder.writeTimestamp(querySet, 1);

		commandEncoder.resolveQuerySet(
			querySet,
			0,
			capacity,
			queryBuffer,
			0
		);

		commandEncoder.copyBufferToBuffer(
			queryBuffer,
			0,
			resultBuffer, 
			0,
			8 * capacity
		);
		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		await resultBuffer.mapAsync(GPUMapMode.READ);
		const timestamps = new BigUint64Array(resultBuffer.getMappedRange());
		const durationNs = Number(timestamps[capacity - 2] - timestamps[0]);
		const durationMs = durationNs / 1_000_000;
		resultBuffer.unmap();
		return durationMs;
	}

	async restrict(level, sourceTexture, targetTexture, isBoundary) {
		const sourceWidth = this.grid[level].width;
		const sourceHeight = this.grid[level].height;
		const targetWidth = this.grid[level + 1].width;
		const targetHeight = this.grid[level + 1].height;

		const boundaryBuffer = this.device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(
			boundaryBuffer,
			0,
			new Uint32Array([isBoundary ? 1 : 0])
		);

		const textureSampler = this.device.createSampler({
			magFilter: 'linear',  
			minFilter: 'linear',  
			addressModeU: 'clamp-to-edge', 
			addressModeV: 'clamp-to-edge',
		});

		const capacity = 3;//Max number of timestamps we can store
		const querySet = this.device.createQuerySet({
			type: "timestamp",
			count: capacity,
		});
		const queryBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage: GPUBufferUsage.QUERY_RESOLVE 
			| GPUBufferUsage.STORAGE
			| GPUBufferUsage.COPY_SRC
			| GPUBufferUsage.COPY_DST,
		});
		const resultBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const commandEncoder = this.device.createCommandEncoder();
		const bindingGroup = this.device.createBindGroup({
			layout: this.restrictionPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: sourceTexture.createView(),
				},
				{
					binding: 1,
					resource: targetTexture.createView(),
				},
				{
					binding: 2,
					resource: textureSampler,
				},
				{
					binding: 3,
					resource: { buffer: boundaryBuffer },
				},
			],
		});

		commandEncoder.writeTimestamp(querySet, 0);
		const pass = commandEncoder.beginComputePass();
		pass.setPipeline(this.restrictionPipeline);
		pass.setBindGroup(0, bindingGroup);
		pass.dispatchWorkgroups(
			Math.ceil(targetWidth / 8),
			Math.ceil(targetHeight / 8)
		);
		pass.end();
		commandEncoder.writeTimestamp(querySet, 1);

		commandEncoder.resolveQuerySet(
			querySet,
			0,
			capacity,
			queryBuffer,
			0
		);

		commandEncoder.copyBufferToBuffer(
			queryBuffer,
			0,
			resultBuffer, 
			0,
			8 * capacity
		);

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		await resultBuffer.mapAsync(GPUMapMode.READ);
		const timestamps = new BigUint64Array(resultBuffer.getMappedRange());
		const durationNs = Number(timestamps[capacity - 2] - timestamps[0]);
		const durationMs = durationNs / 1_000_000;

		resultBuffer.unmap();

		return durationMs;
	}

	async correct(level) {
		const fineLevel = this.grid[level];
		const coarseLevel = this.grid[level + 1];

		const textureSampler = this.device.createSampler({
			magFilter: 'linear',  
			minFilter: 'linear',  
			addressModeU: 'clamp-to-edge', 
			addressModeV: 'clamp-to-edge',
		});

		const capacity = 3;//Max number of timestamps we can store
		const querySet = this.device.createQuerySet({
			type: "timestamp",
			count: capacity,
		});
		const queryBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage: GPUBufferUsage.QUERY_RESOLVE 
			| GPUBufferUsage.STORAGE
			| GPUBufferUsage.COPY_SRC
			| GPUBufferUsage.COPY_DST,
		});
		const resultBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});


		const commandEncoder = this.device.createCommandEncoder();
		const bindingGroup = this.device.createBindGroup({
			layout: this.correctionPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: fineLevel.reconstructionRead.createView(),
				},
				{
					binding: 1,
					resource: coarseLevel.reconstructionRead.createView(),
				},
				{
					binding: 2,
					resource: fineLevel.reconstructionWrite.createView(),
				},
				{
					binding: 3,
					resource: textureSampler,
				},
			],
		});

		commandEncoder.writeTimestamp(querySet, 0);
		const pass = commandEncoder.beginComputePass();
		pass.setPipeline(this.correctionPipeline);
		pass.setBindGroup(0, bindingGroup);
		pass.dispatchWorkgroups(
			Math.ceil(fineLevel.width / 8),
			Math.ceil(fineLevel.height / 8)
		);
		pass.end();
		commandEncoder.writeTimestamp(querySet, 1);

		commandEncoder.resolveQuerySet(
			querySet,
			0,
			capacity,
			queryBuffer,
			0
		);

		commandEncoder.copyBufferToBuffer(
			queryBuffer,
			0,
			resultBuffer, 
			0,
			8 * capacity
		);

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		await resultBuffer.mapAsync(GPUMapMode.READ);
		const timestamps = new BigUint64Array(resultBuffer.getMappedRange());
		const durationNs = Number(timestamps[capacity - 2] - timestamps[0]);
		const durationMs = durationNs / 1_000_000;

		resultBuffer.unmap();

		return durationMs;


		// await this.copyTexture(
		// 	fineLevel.reconstructionWrite,
		// 	fineLevel.reconstructionRead
		// );
	}

	async copyTexture(fromTexture, toTexture) {
		const commandEncoder = this.device.createCommandEncoder();
		commandEncoder.copyTextureToTexture(
			{ texture: fromTexture },
			{ texture: toTexture },
			[
				fromTexture.width || this.width,
				fromTexture.height || this.height,
				1,
			]
		);
		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();
	}

	destroy() {
		for (let i = 0; i < this.levels; i++) {
			this.grid[i].reconstructionRead.destroy();
			this.grid[i].reconstructionWrite.destroy();
			this.grid[i].points.destroy();
			this.grid[i].f.destroy();
			this.grid[i].temp.destroy();
		}
	}
}
