import { PriorityQueue } from "./PriorityQueue.js";
import { saveSDFToPNG, saveTextureToPNG } from "./Utils.js";
import { Solver } from "./Solvers.js";
import { MultigridSolver } from "./MultigridSolver.js";

export class Evaluation {
	constructor(
		device,
		canvas,
		width,
		height,
		maxIterations,
		nSolve,
		nSmooth,
		orig_im,
		rec_im
	) {
		this.device = device;
		this.canvas = canvas;
		this.width = width;
		this.height = height;
		this.maxIterations = maxIterations;
		this.nSolve = nSolve;
		this.nSmooth = nSmooth;
		this.orig_image_file = orig_im;
		this.image_file = rec_im;

		this.querySet = device.createQuerySet({
			type: "timestamp",
			count: 2,
		});

		this.queryBuffer = device.createBuffer({
			size: 2 * 8,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
		});

		this.resultBuffer = device.createBuffer({
			size: 2 * 8,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
	}

	async evaluate_jacobi() {
		const [orig, rec] = await Promise.all(
			[this.orig_image_file, this.image_file].map((url) =>
				fetch(new URL(url, import.meta.url))
					.then((response) => response.blob())
					.then((blob) => createImageBitmap(blob))
			)
		);

		const origTexture = this.device.createTexture({
			size: [orig.width, orig.height],
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
			format: "rgba32float",
		});

		const captureTexture = this.device.createTexture({
			size: [rec.width, rec.height],
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
			format: "rgba32float",
		});

		this.device.queue.copyExternalImageToTexture(
			{ source: orig },
			{
				texture: origTexture,
				origin: [0, 0],
			},
			[orig.width, orig.height]
		);

		this.device.queue.copyExternalImageToTexture(
			{ source: rec },
			{
				texture: captureTexture,
				origin: [0, 0],
			},
			[rec.width, rec.height]
		);

		let reconstructionRead = this.device.createTexture({
			size: [rec.width, rec.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
		});

		this.device.queue.copyExternalImageToTexture(
			{ source: rec },
			{
				texture: reconstructionRead,
				origin: [0, 0],
			},
			[rec.width, rec.height]
		);

		let reconstructionWrite = this.device.createTexture({
			size: [rec.width, rec.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
		});

		let solver = new Solver(this.canvas, this.device);
		solver.width = rec.width;
		solver.height = rec.height;
		solver.maxIterations = this.maxIterations;

		var image = await solver.jacobian(
			captureTexture,
			reconstructionRead,
			reconstructionWrite
		);

		await saveTextureToPNG(
			image,
			rec.width,
			rec.height,
			`${this.image_file}_jacobi_${this.maxIterations}.png`
		);

		let psnr = this.calculatePSNR(
			origTexture,
			image,
			rec.width,
			rec.height
		);
		return psnr;
	}

	async evaluate_sor() {
		const [orig, rec] = await Promise.all(
			[this.orig_image_file, this.image_file].map((url) =>
				fetch(new URL(url, import.meta.url))
					.then((response) => response.blob())
					.then((blob) => createImageBitmap(blob))
			)
		);

		const origTexture = this.device.createTexture({
			size: [orig.width, orig.height],
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
			format: "rgba32float",
		});

		const captureTexture = this.device.createTexture({
			size: [rec.width, rec.height],
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
			format: "rgba32float",
		});

		this.device.queue.copyExternalImageToTexture(
			{ source: orig },
			{
				texture: origTexture,
				origin: [0, 0],
			},
			[orig.width, orig.height]
		);

		this.device.queue.copyExternalImageToTexture(
			{ source: rec },
			{
				texture: captureTexture,
				origin: [0, 0],
			},
			[rec.width, rec.height]
		);

		let reconstructionRead = this.device.createTexture({
			size: [rec.width, rec.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
		});

		this.device.queue.copyExternalImageToTexture(
			{ source: rec },
			{
				texture: reconstructionRead,
				origin: [0, 0],
			},
			[rec.width, rec.height]
		);

		let reconstructionWrite = this.device.createTexture({
			size: [rec.width, rec.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
		});

		let solver = new Solver(this.canvas, this.device);
		solver.width = rec.width;
		solver.height = rec.height;
		solver.maxIterations = this.maxIterations;
		let image = await solver.sorRedBlack(
			captureTexture,
			reconstructionRead,
			reconstructionWrite
		);

		await saveTextureToPNG(
			image,
			rec.width,
			rec.height,
			`${this.image_file}_sor.png`
		);

		let psnr = await this.calculatePSNR(
			origTexture,
			image,
			rec.width,
			rec.height
		);
		return psnr;
	}

	async evaluate_multigrid() {
		const [orig, rec] = await Promise.all(
			[this.orig_image_file, this.image_file].map((url) =>
				fetch(new URL(url, import.meta.url))
					.then((response) => response.blob())
					.then((blob) => createImageBitmap(blob))
			)
		);

		const origTexture = this.device.createTexture({
			size: [orig.width, orig.height],
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
			format: "rgba32float",
		});

		const captureTexture = this.device.createTexture({
			size: [rec.width, rec.height],
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.RENDER_ATTACHMENT,
			format: "rgba32float",
		});

		this.device.queue.copyExternalImageToTexture(
			{ source: orig },
			{
				texture: origTexture,
				origin: [0, 0],
			},
			[orig.width, orig.height]
		);

		this.device.queue.copyExternalImageToTexture(
			{ source: rec },
			{
				texture: captureTexture,
				origin: [0, 0],
			},
			[rec.width, rec.height]
		);

		let multigridSolver = new MultigridSolver(this.canvas, this.device);
		multigridSolver.width = rec.width;
		multigridSolver.height = rec.height;
		multigridSolver.levels = 8;
		multigridSolver.nSolve = this.nSolve;
		multigridSolver.nSmooth = this.nSmooth;

		var image = await multigridSolver.multigridSolve(captureTexture);

		// await saveTextureToPNG(
		// 	image,
		// 	rec.width,
		// 	rec.height,
		// 	`${this.image_file}_multigrid_${this.nSolve}_${this.nSmooth}.png`
		// );

		let psnr = await this.calculatePSNR(
			origTexture,
			image,
			rec.width,
			rec.height
		);
		return psnr;
	}

	async calculatePSNR(originalTexture, reconstructedData, width, height) {
		const bufferSize = width * height * 4 * 4;
		const outputBuffer = this.device.createBuffer({
			size: bufferSize,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const commandEncoder = this.device.createCommandEncoder();
		commandEncoder.copyTextureToBuffer(
			{
				texture: originalTexture,
				mipLevel: 0,
				origin: { x: 0, y: 0, z: 0 },
			},
			{
				buffer: outputBuffer,
				bytesPerRow: width * 16,
				rowsPerImage: height,
			},
			[width, height, 1]
		);

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		await outputBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = outputBuffer.getMappedRange();
		const originalData = new Float32Array(arrayBuffer.slice(0));

		let mse = 0;
		for (let i = 0; i < originalData.length; i++) {
			const diff = originalData[i] * 255 - reconstructedData[i];
			mse += diff * diff;
		}
		mse /= originalData.length;

		const maxPixelValue = 255.0;
		const psnr = 10 * Math.log10((maxPixelValue * maxPixelValue) / mse);

		outputBuffer.unmap();

		return psnr;
	}
}