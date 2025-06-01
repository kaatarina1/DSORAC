// Updated main.js with position padding corrections
import * as mat4 from "./js/mat4.js";
import { PointerController } from "./js/PointerController.js";
import { saveTextureToPNG, convertTexture } from "./js/Utils.js";
import { Solver } from "./js/Solvers.js";
import { DepthMap } from "./js/DepthMap.js";
import { Composer } from "./js/Composer.js";
import { LasLoader } from "./js/LasLoader.js";
import { MultigridSolver } from "./js/MultigridSolver.js";
import { ConjugateGradientSolver } from "./js/ConjugateGradientSolver.js";
import { SignedDistanceFiled } from "./js/SignedDistanceFiled.js";
import { Evaluation } from "./js/Evaluation.js";

const adapter = await navigator.gpu.requestAdapter();
const hasBGRA8unormStorage = adapter.features.has("bgra8unorm-storage");
const device = await adapter?.requestDevice({
	requiredFeatures: hasBGRA8unormStorage ? ["bgra8unorm-storage", "float32-filterable", "float32-blendable"] : ["float32-filterable", "float32-blendable"],
});
const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");
const format = hasBGRA8unormStorage
	? navigator.gpu.getPreferredCanvasFormat()
	: "rgba8unorm"; // TODO: format za shranjevanje slik rgb 32 float, -> slike shranjevat v pam
// const imageFormat = "rgba32float";
context.configure({ device, format, alphaMode: "premultiplied" });

const composer = new Composer(document, canvas, device, format);
composer.initializeBackground("./data/sky.jpg");

const renderingCode = await fetch("./shaders/rendering.wgsl").then((response) =>
	response.text()
);

const renderingModule = device.createShaderModule({ code: renderingCode });

const renderingPipeline = device.createRenderPipeline({
	vertex: {
		module: renderingModule,
	},
	fragment: {
		module: renderingModule,
		targets: [{ format }],
	},
	primitive: {
		topology: "point-list",
	},
	depthStencil: {
		depthWriteEnabled: true,
		depthCompare: "less",
		format: "depth32float",
	},
	layout: "auto",
});

const lasFile = `./data/cropped_filtered_1.las`;

const lasLoader = new LasLoader(lasFile);
const lasData = await lasLoader.loadLasData();
const positions = lasData.positions;
const colors = lasData.colors;

const numberOfAllPoints = positions.length / 3;
const maxNumberOfPointsPerBuffer = 1024 * 1024;
const pointByteSize = 16;

console.log(numberOfAllPoints);
console.log(numberOfAllPoints / maxNumberOfPointsPerBuffer);

function createPointCloud(numberOfPoints, count) {
	if (numberOfPoints > maxNumberOfPointsPerBuffer) {
		throw new Error("Too many points for one point buffer");
	}

	const pointBuffer = device.createBuffer({
		size: numberOfPoints * pointByteSize,
		usage:
			GPUBufferUsage.STORAGE |
			GPUBufferUsage.COPY_DST |
			GPUBufferUsage.COPY_SRC,
	});

	// Create a CPU-side point array
	const pointData = new ArrayBuffer(numberOfPoints * pointByteSize);
	const pointDataView = new DataView(pointData);

	// Populate the data with positions and colors
	let startIndex = count * numberOfPoints;
	for (let i = 0; i < numberOfPoints; i++) {
		const posIndex = (startIndex + i) * 3;
		const pointOffset = i * pointByteSize;

		// Write position XYZ
		pointDataView.setFloat32(pointOffset, positions[posIndex], true);
		pointDataView.setFloat32(
			pointOffset + 4,
			positions[posIndex + 1],
			true
		);
		pointDataView.setFloat32(
			pointOffset + 8,
			positions[posIndex + 2],
			true
		);

		// Write color as uint32
		pointDataView.setUint32(pointOffset + 12, colors[startIndex + i], true);
	}

	// Write the point data directly to the GPU buffer
	device.queue.writeBuffer(pointBuffer, 0, pointData);

	const uniformBuffer = device.createBuffer({
		size: 4,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	device.queue.writeBuffer(
		uniformBuffer,
		0,
		new Uint32Array([numberOfPoints])
	);

	const renderingBindGroup = device.createBindGroup({
		layout: renderingPipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: { buffer: pointBuffer },
			},
		],
	});

	return {
		pointBuffer,
		uniformBuffer,
		renderingBindGroup,
		numberOfPoints,
	};
}

const pointclouds = [];
for (let i = 0; i < numberOfAllPoints / maxNumberOfPointsPerBuffer; i++) {
	const pointcloud = createPointCloud(
		numberOfAllPoints > maxNumberOfPointsPerBuffer
			? maxNumberOfPointsPerBuffer
			: numberOfAllPoints,
		i
	);
	pointclouds.push(pointcloud);
	// For debugging if needed
	// const stagingBuffer = device.createBuffer({
	//     size: pointcloud.numberOfPoints * pointByteSize,
	//     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	// });
	// const commandEncoder = device.createCommandEncoder();
	// commandEncoder.copyBufferToBuffer(
	//     pointcloud.pointBuffer, // Source buffer
	//     0, // Source offset
	//     stagingBuffer, // Destination buffer
	//     0, // Destination offset
	//     pointcloud.numberOfPoints * pointByteSize // Size of data to copy
	// );
	// device.queue.submit([commandEncoder.finish()]);
	// readPointBuffer(stagingBuffer);
}

// Used for debugging
async function readPointBuffer(stagingBuffer) {
	// Ensure the buffer is mapped for reading
	await stagingBuffer.mapAsync(GPUMapMode.READ);

	// Access the buffer data
	const arrayBuffer = stagingBuffer.getMappedRange();

	// Interpret the buffer data
	const pointData = new DataView(arrayBuffer);

	for (let i = 0; i < 3; i++) {
		const baseOffset = i * pointByteSize; // Assuming each point is 24 bytes
		const x = pointData.getFloat32(baseOffset, true); // Read position.x
		const y = pointData.getFloat32(baseOffset + 4, true); // Read position.y
		const z = pointData.getFloat32(baseOffset + 8, true); // Read position.z
		const color = pointData.getUint32(baseOffset + 12, true); // Read packed color

		console.log(
			`Point ${i}: Position = (${x}, ${y}, ${z}), Color = 0x${color.toString(
				16
			)}`
		);
	}

	// Unmap the buffer
	stagingBuffer.unmap();
}

const parameterBuffer = device.createBuffer({
	size: 4,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const matrixBuffer = device.createBuffer({
	size: 64,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const matrixBindGroup = device.createBindGroup({
	layout: renderingPipeline.getBindGroupLayout(1),
	entries: [
		{
			binding: 0,
			resource: { buffer: matrixBuffer },
		},
	],
});

let depthTexture = device.createTexture({
	size: [canvas.width, canvas.height],
	usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
	format: "depth32float",
});

let lastSize = { width: null, height: null };

let horizontalAngle = 1.12; // Angle for horizontal rotation (around Z-axis)
let verticalAngle = -1.6; // Angle for vertical rotation (around X-axis)
let moveX = 0.09; // Left-right movement
let moveY = 0.12; // Up-down movement
let zoomFactor = 3.0; // Default zoom factor

const pointerController = new PointerController();

// Mouse drag behavior (rotating camera)
pointerController.addEventListener("pointermove", (e) => {
	const dx = e.movementX;
	const dy = e.movementY;

	const sensitivity = 0.005; // Adjust this value for more or less sensitivity

	if (e.buttons === 1) {
		// Left button (rotate)
		// Horizontal movement (left-right) rotates around the Z-axis
		horizontalAngle -= dx * sensitivity;

		// Vertical movement (up-down) rotates around the X-axis
		verticalAngle -= dy * sensitivity;

		// Prevent excessive vertical rotation (clamping to avoid gimbal lock)
		verticalAngle = Math.max(
			-Math.PI / 2,
			Math.min(Math.PI / 2, verticalAngle)
		);
	} else if (e.buttons === 2) {
		// Right button (move left-right and up-down)
		moveX -= dx * sensitivity;
		moveY += dy * sensitivity;
	}
});

// Zoom behavior (scroll)
pointerController.addEventListener("wheel", (e) => {
	e.preventDefault(); // Prevent default scroll behavior to allow zoom
	const delta = e.deltaY; // Positive value for scrolling down, negative for up

	// Zoom in/out based on scroll direction
	if (delta > 0) {
		zoomFactor *= 1.05; // Zoom in
	} else {
		zoomFactor /= 1.05; // Zoom out
	}
});

document.addEventListener("keydown", async (event) => {
	if (event.key === "T" || event.key === "t") {
		const startTime = performance.now();
		console.log("Capturing the current view...");

		composer.initializeCompositeTexture();

		let depthMap = new DepthMap(canvas, device, depthTexture);
		let depthValues = await depthMap.groupDepthIntoBins();
		// depthValues.reverse();

		for (let i = 0; i < 6; i++) {
			console.log(i);
			await renderPointsInDepthRange(
				depthValues[i][0],
				depthValues[i][1]
			);
		}

		await composer.compositeDepths();

		// await renderPointsInDepthRange(
		// 	0,
		// 	1
		// );

		const endTime = performance.now();
		console.log(`Composite completed in ${endTime - startTime} ms`);
	}
});

document.addEventListener("keydown", async (event) => {
	if (event.key === "E" || event.key === "e") {
		const startTime = performance.now();

		let orig_images = [
			"../data/images/image2.png",
			"../data/images/image3.png",
		];
		let rec_images = [
			"../data/images/image2_50_alpha.png",
			"../data/images/image3_50_alpha.png",
			"../data/images/image2_80_alpha.png",
			"../data/images/image3_80_alpha.png",
			"../data/images/image2_90_alpha.png",
			"../data/images/image3_90_alpha.png",
			"../data/images/image2_95_alpha.png",
			"../data/images/image3_95_alpha.png",
			"../data/images/image2_99_alpha.png",
			"../data/images/image3_99_alpha.png",
		];
		let maxIterations = [10, 50, 100, 500, 1000, 2000, 3000, 3500, 5000];

		// console.log("--------------- Jacobi --------------------");
		// for (let j = 0; j < rec_images.length; j++) {
		// 	console.log("IMAGE ", rec_images[j]);
		// 	for (let i = 0; i < maxIterations.length; i++) {
		// 		console.log(
		// 			"Reconstruction evaluation for ",
		// 			maxIterations[i],
		// 			"iterations"
		// 		);
		// 		const evaluator = new Evaluation(
		// 			device,
		// 			canvas,
		// 			canvas.width,
		// 			canvas.height,
		// 			maxIterations[i],
		// 			0,
		// 			0,
		// 			orig_images[j % 2],
		// 			rec_images[j]
		// 		);
		// 		let psnr = await evaluator.evaluate_jacobi();
		// 		console.log("PSNR = ", psnr);
		// 	}
		// }

		// console.log("--------------- SOR --------------------");
		// for (let j = 0; j < rec_images.length; j++) {
		// 	console.log("IMAGE ", rec_images[j]);
		// 	for (let i = 0; i < maxIterations.length; i++) {
		// 		console.log(
		// 			"Reconstruction evaluation for ",
		// 			maxIterations[i],
		// 			"iterations"
		// 		);
		// 		const evaluator = new Evaluation(
		// 			device,
		// 			canvas,
		// 			canvas.width,
		// 			canvas.height,
		// 			maxIterations[i],
		// 			0,
		// 			0,
		// 			orig_images[j % 2],
		// 			rec_images[j]
		// 		);
		// 		let psnr = await evaluator.evaluate_sor();
		// 		console.log("PSNR = ", psnr);
		// 	}
		// }

		let nSoves = [2, 5, 10, 20];
		let nSmooths = [5, 10, 20, 50];
		for (let j = 0; j < rec_images.length; j++) {
			console.log("IMAGE ", rec_images[j]);
			for (let i = 0; i < nSoves.length; i++) {
				console.log(
					"Reconstruction evaluation for ",
					nSoves[i],
					" nSolves"
				);
				for (let k = 0; k < nSmooths.length; k++) {
					console.log(
						"Reconstruction evaluation for ",
						nSmooths[k],
						" nSmooth"
					);
					const evaluator = new Evaluation(
						device,
						canvas,
						canvas.width,
						canvas.height,
						0,
						nSoves[i],
						nSmooths[k],
						orig_images[j % 2],
						rec_images[j]
					);
					let psnr = await evaluator.evaluate_multigrid();
					console.log("PSNR = ", psnr);
				}
			}
		}

		// const evaluator = new Evaluation(
		// 	device,
		// 	canvas,
		// 	canvas.width,
		// 	canvas.height,
		// 	maxIterations[5],
		// 	nSoves[2],
		// 	nSmooths[3],
		// 	orig_images[0],
		// 	rec_images[0]
		// );
		// let psnr = await evaluator.evaluate_jacobi();
		// let psnr2 = await evaluator.evaluate_sor();
		// await evaluator.evaluate_multigrid();
	}
});

async function renderPointsInDepthRange(minDepth, maxDepth) {
	// Create a buffer for depth range
	let depthRangeBuffer = device.createBuffer({
		size: 8, // Two floats (4 bytes each)
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	// Write the min and max depth to the buffer
	device.queue.writeBuffer(
		depthRangeBuffer,
		0,
		new Float32Array([minDepth, maxDepth])
	);

	// Create a bind group for depth filtering
	let depthRangeBindGroup = device.createBindGroup({
		layout: renderingPipeline.getBindGroupLayout(2),
		entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
	});

	const captureTexture = device.createTexture({
		size: [canvas.width, canvas.height],
		usage:
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_SRC |
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.RENDER_ATTACHMENT,
		format: format,
	});

	// Render with updated depth filtering
	const commandEncoder = device.createCommandEncoder();
	const renderPass = commandEncoder.beginRenderPass({
		colorAttachments: [
			{
				view: captureTexture.createView(),
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			},
		],
		depthStencilAttachment: {
			view: depthTexture.createView(),
			depthLoadOp: "clear",
			depthClearValue: 1,
			depthStoreOp: "store",
		},
	});

	renderPass.setPipeline(renderingPipeline);
	renderPass.setBindGroup(1, matrixBindGroup);
	renderPass.setBindGroup(2, depthRangeBindGroup); // Apply depth range filter

	for (const pointcloud of pointclouds) {
		renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
		renderPass.draw(pointcloud.numberOfPoints);
	}

	renderPass.end();

	// Copy the texture content into a buffer
	const outputBuffer = device.createBuffer({
		size: canvas.width * canvas.height * 4, // 4 bytes per pixel (RGBA)
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	commandEncoder.copyTextureToBuffer(
		{
			texture: captureTexture,
			mipLevel: 0,
			origin: { x: 0, y: 0, z: 0 },
		},
		{
			buffer: outputBuffer,
			bytesPerRow: canvas.width * 4,
			rowsPerImage: canvas.height,
		},
		[canvas.width, canvas.height, 1]
	);

	// Create texture resources
	let reconstructionRead = device.createTexture({
		size: [canvas.width, canvas.height],
		format: "rgba32float",
		usage:
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
	});

	await convertTexture(
		device,
		canvas.width,
		canvas.height,
		captureTexture,
		reconstructionRead
	);

	let reconstructionWrite = device.createTexture({
		size: [canvas.width, canvas.height],
		format: "rgba32float",
		usage:
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
	});
	device.queue.submit([commandEncoder.finish()]);

	// Read the buffer
	await outputBuffer.mapAsync(GPUMapMode.READ);
	const arrayBuffer = outputBuffer.getMappedRange();
	const imageData = new Uint8Array(arrayBuffer);

	// await saveTextureToPNG(
	// 	imageData,
	// 	canvas.width,
	// 	canvas.height,
	// 	`orig_depth_layer_${minDepth}_${maxDepth}.png`
	// );

	const sdf = new SignedDistanceFiled(
		device,
		captureTexture,
		canvas.width,
		canvas.height
	);
	const sdfTexture = await sdf.generateSDF();

	// var image;
	// for (let i = 0; i < 60; i++) {
	// 	let solver = new Solver(canvas, device);
	// 	image = await solver.jacobian(
	// 		captureTexture,
	// 		reconstructionRead,
	// 		reconstructionWrite
	// 	);
	// }

	// let solver = new Solver(canvas, device);
	// let image = await solver.sorRedBlack(
	// 	captureTexture,
	// 	reconstructionRead,
	// 	reconstructionWrite
	// );

	let multigridSolver = new MultigridSolver(canvas, device);
	let image = await multigridSolver.multigridSolve(captureTexture);

	// let cgSolver = new ConjugateGradientSolver(canvas, device);
	// let image = await cgSolver.conjGradientSolve(captureTexture, reconstructionRead, reconstructionWrite);

	// await saveTextureToPNG(
	// 	image,
	// 	canvas.width,
	// 	canvas.height,
	// 	`depth_layer_${minDepth}_${maxDepth}.png`
	// );

	let pointsTexture = device.createTexture({
		size: [canvas.width, canvas.height],
		format: "rgba32float",
		usage:
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
	});

	await convertTexture(
		device,
		canvas.width,
		canvas.height,
		captureTexture,
		pointsTexture
	);

	await composer.addLayers(sdfTexture, reconstructionRead, pointsTexture, (minDepth + maxDepth) / 2);

	captureTexture.destroy();
	reconstructionRead.destroy();
	reconstructionWrite.destroy();
	sdfTexture.destroy();
	depthRangeBuffer.destroy();
}

function frame() {
	// Resize
	const size = canvas.getBoundingClientRect();
	if (size.width !== lastSize.width || size.height !== lastSize.height) {
		canvas.width = lastSize.width = size.width;
		canvas.height = lastSize.height = size.height;
		depthTexture.destroy();
		depthTexture = device.createTexture({
			size: [canvas.width, canvas.height],
			usage:
				GPUTextureUsage.RENDER_ATTACHMENT |
				GPUTextureUsage.TEXTURE_BINDING,
			format: "depth32float",
		});
	}

	device.queue.writeBuffer(parameterBuffer, 0, new Float32Array([1 / 60]));

	const commandEncoder = device.createCommandEncoder();

	// Calculate the rotation matrix based on horizontal and vertical angles
	const modelMatrix = mat4.multiply(
		mat4.rotateX(verticalAngle), // Rotate around X-axis (up-down)
		mat4.rotateZ(horizontalAngle) // Rotate around Z-axis (left-right)
	);

	// Apply the position adjustments from the right-click drag (left-right, up-down)
	const viewMatrix = mat4.translation([moveX, moveY, -0.1 * zoomFactor]);

	const projectionMatrix = mat4.perspective(
		1,
		canvas.width / canvas.height,
		0.1,
		100
	);
	const matrix = mat4.multiply(projectionMatrix, viewMatrix, modelMatrix);

	device.queue.writeBuffer(matrixBuffer, 0, matrix);

	const renderPass = commandEncoder.beginRenderPass({
		colorAttachments: [
			{
				view: context.getCurrentTexture().createView(),
				loadOp: "clear",
				clearValue: [0, 0, 0, 0],
				storeOp: "store",
			},
		],
		depthStencilAttachment: {
			view: depthTexture.createView(),
			depthLoadOp: "clear",
			depthClearValue: 1,
			depthStoreOp: "store",
		},
	});

	// Create a buffer for depth range
	let depthRangeBuffer = device.createBuffer({
		size: 8, // Two floats (4 bytes each)
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	// Write the min and max depth to the buffer
	device.queue.writeBuffer(depthRangeBuffer, 0, new Float32Array([0, 1]));

	// Create a bind group for depth filtering
	let depthRangeBindGroup = device.createBindGroup({
		layout: renderingPipeline.getBindGroupLayout(2),
		entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
	});
	renderPass.setPipeline(renderingPipeline);
	renderPass.setBindGroup(1, matrixBindGroup);
	renderPass.setBindGroup(2, depthRangeBindGroup); // Apply depth range filter

	for (const pointcloud of pointclouds) {
		renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
		renderPass.draw(pointcloud.numberOfPoints);
	}
	renderPass.end();

	device.queue.submit([commandEncoder.finish()]);

	requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
