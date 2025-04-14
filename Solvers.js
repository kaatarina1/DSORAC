export class Solver {
	constructor(canvas, device, pipeline) {
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
		this.pipeline = pipeline;
	}

	async jacobian(captureTexture, reconstructionRead, reconstructionWrite) {
		// Perform Jacobian operation
		const jacobiCommandEncoder = this.device.createCommandEncoder();

		const numOfIter = 100000;
		for (let i = 0; i < numOfIter; i++) {
			const jacobiaBindGroup = this.device.createBindGroup({
				layout: this.pipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: captureTexture.createView() },
					{ binding: 1, resource: reconstructionRead.createView() },
					{ binding: 2, resource: reconstructionWrite.createView() },
				],
			});
			const jacobiaPassEncoder = jacobiCommandEncoder.beginComputePass();
			jacobiaPassEncoder.setPipeline(this.pipeline);
			jacobiaPassEncoder.setBindGroup(0, jacobiaBindGroup);
			jacobiaPassEncoder.dispatchWorkgroups(
				Math.ceil(this.width / 8),
				Math.ceil(this.height / 8)
			);
			jacobiaPassEncoder.end();

			[reconstructionRead, reconstructionWrite] = [
				reconstructionWrite,
				reconstructionRead,
			];
		}

		const jacobiaOutputBuffer = this.device.createBuffer({
			size: this.width * this.height * 4, // 4 bytes per pixel (RGBA)
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
				bytesPerRow: this.width * 4,
				rowsPerImage: this.height,
			},
			[this.width, this.height, 1]
		);

		this.device.queue.submit([jacobiCommandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		// Read back and save the Jacobian results
		await jacobiaOutputBuffer.mapAsync(GPUMapMode.READ);
		const jacobiaArrayBuffer = jacobiaOutputBuffer.getMappedRange();
		const jacobianImage = new Uint8Array(jacobiaArrayBuffer.slice(0));

		console.log("Expected buffer size:", this.width * this.height * 4);
		console.log("Actual buffer size:", jacobiaOutputBuffer.size);
		console.log("ArrayBuffer length:", jacobiaArrayBuffer.byteLength);
		console.log("Is buffer mapped?", jacobiaOutputBuffer.mapState);
		jacobiaOutputBuffer.unmap();

		return jacobianImage;
	}

	// Additional solver methods like SOR, Gauss-Seidel, etc. can be added here
	async sorRedBlack(captureTexture, reconstructionRead, reconstructionWrite) {
		// Perform reconstruction using updateRedBlack on the original input

		const redBlackBuffer = this.device.createBuffer({
			size: 4,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const redBlackCommandEncoder = this.device.createCommandEncoder();
		for (let color = 0; color < 200000; color++) {
			this.device.queue.writeBuffer(
				redBlackBuffer,
				0,
				new Uint32Array([color % 2])
			);

			const updateBindGroup = this.device.createBindGroup({
				layout: this.pipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: captureTexture.createView() },
					{ binding: 1, resource: reconstructionRead.createView() },
					{ binding: 2, resource: reconstructionWrite.createView() },
					{ binding: 3, resource: { buffer: redBlackBuffer } },
				],
			});

			const redBlackPass = redBlackCommandEncoder.beginComputePass();
			redBlackPass.setPipeline(this.pipeline);
			redBlackPass.setBindGroup(0, updateBindGroup);
			redBlackPass.dispatchWorkgroups(
				Math.ceil(this.width / 8),
				Math.ceil(this.height / 8)
			);
			redBlackPass.end();

			[reconstructionRead, reconstructionWrite] = [
				reconstructionWrite,
				reconstructionRead,
			];
		}

		const sorOutputBuffer = this.device.createBuffer({
			size: this.width * this.height * 4, // 4 bytes per pixel (RGBA)
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		redBlackCommandEncoder.copyTextureToBuffer(
			{
				texture: reconstructionRead,
				mipLevel: 0,
				origin: { x: 0, y: 0, z: 0 },
			},
			{
				buffer: sorOutputBuffer,
				bytesPerRow: this.width * 4,
				rowsPerImage: this.height,
			},
			[this.width, this.height, 1]
		);

		this.device.queue.submit([redBlackCommandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		// Read back and save the Jacobian results
		await sorOutputBuffer.mapAsync(GPUMapMode.READ);
		const sorArrayBuffer = sorOutputBuffer.getMappedRange();
		const sornImage = new Uint8Array(sorArrayBuffer);

		return sornImage;
	}
}
