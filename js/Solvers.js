export class Solver {
	constructor(canvas, device) {
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
		this.jacobianPipeline = null;
		this.residualPipeline = null;
		this.updateRedBlackPipeline = null;
		this.maxIterations = 5000;
	}

	async createJacobianPipeline() {
		const jacobiaCode = await fetch("./shaders/jacobia.wgsl").then(
			(response) => response.text()
		);

		const jacobiaModule = this.device.createShaderModule({
			code: jacobiaCode,
		});

		const jacobiaPipeline = this.device.createComputePipeline({
			compute: {
				module: jacobiaModule,
				entryPoint: "main",
			},
			layout: "auto",
		});

		return jacobiaPipeline;
	}

	async createUpdateRedBlackPipeline() {
		const updateRedBlackWGSL = await fetch(
			"./shaders/updateRedBlack.wgsl"
		).then((response) => response.text());

		const updateRedBlackModule = this.device.createShaderModule({
			code: updateRedBlackWGSL,
		});

		const updateRedBlackPipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: updateRedBlackModule,
				entryPoint: "main",
			},
		});

		return updateRedBlackPipeline;
	}

	async createResidualPipeline() {
		const residualCode = await fetch("./shaders/residual.wgsl").then(
			(response) => response.text()
		);
		const residualModule = this.device.createShaderModule({
			code: residualCode,
		});

		const residualPipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: residualModule,
				entryPoint: "main",
			},
		});

		return residualPipeline;
	}

	async jacobian(captureTexture, reconstructionRead, reconstructionWrite) {
		// Perform Jacobian operation
		this.jacobianPipeline = await this.createJacobianPipeline();
		const jacobiCommandEncoder = this.device.createCommandEncoder();

		const textureSampler = this.device.createSampler({
			magFilter: 'linear',  
			minFilter: 'linear',  
			addressModeU: 'clamp-to-edge', 
			addressModeV: 'clamp-to-edge',
		});

		const capacity = 3; //Max number of timestamps we can store
		const querySet = this.device.createQuerySet({
			type: "timestamp",
			count: capacity,
		});
		const queryBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage:
				GPUBufferUsage.QUERY_RESOLVE |
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
		});
		const resultBuffer = this.device.createBuffer({
			size: 8 * capacity,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		for (let i = 0; i < this.maxIterations; i++) {
			const jacobiaBindGroup = this.device.createBindGroup({
				layout: this.jacobianPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: captureTexture.createView() },
					{ binding: 1, resource: reconstructionRead.createView() },
					{ binding: 2, resource: textureSampler },
					{ binding: 3, resource: reconstructionWrite.createView() },
				],
			});

			if (i == 0) {
				jacobiCommandEncoder.writeTimestamp(querySet, 0);
			}

			const jacobiaPassEncoder = jacobiCommandEncoder.beginComputePass();
			jacobiaPassEncoder.setPipeline(this.jacobianPipeline);
			jacobiaPassEncoder.setBindGroup(0, jacobiaBindGroup);
			jacobiaPassEncoder.dispatchWorkgroups(
				Math.ceil(this.width / 8),
				Math.ceil(this.height / 8)
			);
			jacobiaPassEncoder.end();

			if (i == this.maxIterations - 1) {
				jacobiCommandEncoder.writeTimestamp(querySet, 1);
			}

			[reconstructionRead, reconstructionWrite] = [
				reconstructionWrite,
				reconstructionRead,
			];
		}

		const jacobiaOutputBuffer = this.device.createBuffer({
			size: this.width * this.height * 16,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		jacobiCommandEncoder.copyTextureToBuffer(
			{
				texture: reconstructionRead,
				mipLevel: 0,
				origin: { x: 0, y: 0, z: 0 },
			},
			{
				buffer: jacobiaOutputBuffer,
				bytesPerRow: this.width * 16,
				rowsPerImage: this.height,
			},
			[this.width, this.height, 1]
		);

		jacobiCommandEncoder.resolveQuerySet(
			querySet,
			0,
			capacity,
			queryBuffer,
			0
		);

		jacobiCommandEncoder.copyBufferToBuffer(
			queryBuffer,
			0,
			resultBuffer,
			0,
			8 * capacity
		);

		this.device.queue.submit([jacobiCommandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		// Read back and save the Jacobian results
		await jacobiaOutputBuffer.mapAsync(GPUMapMode.READ);
		const jacobiaArrayBuffer = jacobiaOutputBuffer.getMappedRange();
		const floatData = new Float32Array(jacobiaArrayBuffer.slice(0));

		// Convert to Uint8Array for display (optional)
		const jacobianImage = new Uint8Array(this.width * this.height * 4);
		for (let i = 0; i < floatData.length; i += 4) {
			// Simple normalization for display
			jacobianImage[i] = Math.max(
				0,
				Math.min(255, floatData[i] * 255)
			);
			jacobianImage[i + 1] = Math.max(
				0,
				Math.min(255, floatData[i + 1] * 255)
			);
			jacobianImage[i + 2] = Math.max(
				0,
				Math.min(255, floatData[i + 2] * 255)
			);
			jacobianImage[i + 3] = Math.max(
				0,
				Math.min(255, floatData[i + 3] * 255)
			);
		}

		jacobiaOutputBuffer.unmap();

		await resultBuffer.mapAsync(GPUMapMode.READ);
		const timestamps = new BigUint64Array(resultBuffer.getMappedRange());
		const durationNs = Number(timestamps[capacity - 2] - timestamps[0]);
		const durationMs = durationNs / 1_000_000;
		console.log("Jacobi duration: ", durationMs);
		console.log("Jacobi duration average per iteration: ", durationMs / this.maxIterations);

		resultBuffer.unmap();
		jacobiaOutputBuffer.unmap();

		return jacobianImage;
	}

	async sorRedBlack(captureTexture, reconstructionRead, reconstructionWrite, f) {
		this.updateRedBlackPipeline = await this.createUpdateRedBlackPipeline();
		const omega = 1.9;

		if (!f) {
			f = this.device.createTexture({
				size: [this.width, this.height],
				format: "rgba32float",
				usage:
					GPUTextureUsage.STORAGE_BINDING |
					GPUTextureUsage.TEXTURE_BINDING |
					GPUTextureUsage.COPY_SRC,
				});
		}

		const textureSampler = this.device.createSampler({
			magFilter: 'linear',  
			minFilter: 'linear',  
			addressModeU: 'clamp-to-edge', 
			addressModeV: 'clamp-to-edge',
		});

		const redBlackBuffer = this.device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const omegaBuffer = this.device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(
			omegaBuffer,
			0,
			new Float32Array([omega])
		);

		let duration = 0;
		for (let iteration = 0; iteration < this.maxIterations; iteration++) {
			for (let color = 0; color < 2; color++) {
				this.device.queue.writeBuffer(
					redBlackBuffer,
					0,
					new Uint32Array([color % 2])
				);

				const capacity = 3; //Max number of timestamps we can store
				const querySet = this.device.createQuerySet({
					type: "timestamp",
					count: capacity,
				});
				const queryBuffer = this.device.createBuffer({
					size: 8 * capacity,
					usage:
						GPUBufferUsage.QUERY_RESOLVE |
						GPUBufferUsage.STORAGE |
						GPUBufferUsage.COPY_SRC |
						GPUBufferUsage.COPY_DST,
				});
				const resultBuffer = this.device.createBuffer({
					size: 8 * capacity,
					usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
				});

				const commandEncoder = this.device.createCommandEncoder();

				// SOR update pass
				const updateBindGroup = this.device.createBindGroup({
					layout: this.updateRedBlackPipeline.getBindGroupLayout(0),
					entries: [
						{ binding: 0, resource: captureTexture.createView() },
						{
							binding: 1,
							resource: reconstructionRead.createView(),
						},
						{
							binding: 2,
							resource: reconstructionWrite.createView(),
						},
						{
							binding: 3,
							resource: f.createView(),
						},
						{
							binding: 4,
							resource: textureSampler,
						},
						{ binding: 5, resource: { buffer: redBlackBuffer } },
						{ binding: 6, resource: { buffer: omegaBuffer } },
					],
				});

				commandEncoder.writeTimestamp(querySet, 0);
				const updatePass = commandEncoder.beginComputePass();
				updatePass.setPipeline(this.updateRedBlackPipeline);
				updatePass.setBindGroup(0, updateBindGroup);
				updatePass.dispatchWorkgroups(
					Math.ceil(this.width / 8),
					Math.ceil(this.height / 8)
				);
				updatePass.end();

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

				await resultBuffer.mapAsync(GPUMapMode.READ);
				const timestamps = new BigUint64Array(
					resultBuffer.getMappedRange()
				);
				const durationNs = Number(
					timestamps[capacity - 2] - timestamps[0]
				);
				const durationMs = durationNs / 1_000_000;
				duration += durationMs;
				resultBuffer.unmap();

				// Swap textures for next iteration
				[reconstructionRead, reconstructionWrite] = [
					reconstructionWrite,
					reconstructionRead,
				];
			}

			// Optional: Add progress logging
			if (iteration % 100 === 0) {
				console.log(`Completed ${iteration} iterations`);
			}
		}

		const outputBuffer = this.device.createBuffer({
			size: this.width * this.height * 16,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const commandEncoder = this.device.createCommandEncoder();
		commandEncoder.copyTextureToBuffer(
			{
				texture: reconstructionRead,
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
		const arrayBuffer = outputBuffer.getMappedRange();
		const floatData = new Float32Array(arrayBuffer.slice(0));

		const resultImage = new Uint8Array(this.width * this.height * 4);
		for (let i = 0; i < floatData.length; i += 4) {
			resultImage[i] = Math.max(0, Math.min(255, floatData[i] * 255));
			resultImage[i + 1] = Math.max(
				0,
				Math.min(255, floatData[i + 1] * 255)
			);
			resultImage[i + 2] = Math.max(0, Math.min(255, floatData[i + 2] * 255));
			resultImage[i + 3] = Math.max(
				0,
				Math.min(255, floatData[i + 3] * 255)
			);
		}

		outputBuffer.unmap();

		// console.log("SOR time: ", duration);
		// console.log("SOR average time per iteration", duration / this.maxIterations);

		return resultImage;
	}
}
