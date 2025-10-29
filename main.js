// Updated main.js with position padding corrections
import * as mat4 from "./js/mat4.js";
import { PointerController } from "./js/PointerController.js";
import { getBlob, convertTexture, saveTextureToPNG } from "./js/Utils.js";
import { Solver } from "./js/Solvers.js";
import { DepthMap } from "./js/DepthMap.js";
import { Composer } from "./js/Composer.js";
import { LasLoader } from "./js/LasLoader.js";
import { MultigridSolver } from "./js/MultigridSolver.js";
import { ConjugateGradientSolver } from "./js/ConjugateGradientSolver.js";
import { SignedDistanceFiled } from "./js/SignedDistanceFiled.js";
import { Evaluation } from "./js/Evaluation.js";
import { CameraPosition } from "./js/CameraPosition.js";
import JSZip from "jszip";
import saveAs from "file-saver";

const adapter = await navigator.gpu.requestAdapter();
const hasBGRA8unormStorage = adapter.features.has("bgra8unorm-storage");
const device = await adapter?.requestDevice({
	requiredFeatures: hasBGRA8unormStorage ? ["bgra8unorm-storage", "float32-filterable", "float32-blendable", "timestamp-query"] : ["float32-filterable", "float32-blendable", "timestamp-query"],
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
const colorsRGB = lasData.colorsRGB;
const scaleFactor = lasData.scaleFactor;

const numberOfAllPoints = positions.length / 3;
const maxNumberOfPointsPerBuffer = 1024 * 1024;
const pointByteSize = 16;

console.log(numberOfAllPoints);
console.log(numberOfAllPoints / maxNumberOfPointsPerBuffer);

const camPositionHelper = new CameraPosition(canvas, device, renderingPipeline);

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
	const stagingBuffer = device.createBuffer({
	    size: pointcloud.numberOfPoints * pointByteSize,
	    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});
	const commandEncoder = device.createCommandEncoder();
	commandEncoder.copyBufferToBuffer(
	    pointcloud.pointBuffer, // Source buffer
	    0, // Source offset
	    stagingBuffer, // Destination buffer
	    0, // Destination offset
	    pointcloud.numberOfPoints * pointByteSize // Size of data to copy
	);
	device.queue.submit([commandEncoder.finish()]);
	readPointBuffer(stagingBuffer);
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


const pointerController = new PointerController();
// Replace your camera variables with these:
let cameraPosition = [0, 0, 0];
let cameraTarget = [0, -0.17, 0];
let cameraUp = [0, 1, 0]; // Y is up
let yaw = 0; // Horizontal rotation
let pitch = 0; // Looking downward
let distance = 0.2; // Distance from target

document.addEventListener("keydown", (event) => {
    const moveSpeed = 0.01;
    
    // Calculate forward direction (from camera to target) in X-Z plane
    const forward = mat4.normalize([
        cameraTarget[0] - cameraPosition[0],
		0,
        cameraTarget[2] - cameraPosition[2],
    ]);
    
    // Calculate right direction (perpendicular to forward and up)
    const right = mat4.cross(forward, [0, 1, 0]);
    
    switch(event.key) {
        case "ArrowLeft":
            // Strafe left
            cameraPosition[0] -= right[0] * moveSpeed;
            cameraPosition[2] -= right[2] * moveSpeed;
            cameraTarget[0] -= right[0] * moveSpeed;
            cameraTarget[2] -= right[2] * moveSpeed;
            break;
        case "ArrowRight":
            // Strafe right
            cameraPosition[0] += right[0] * moveSpeed;
            cameraPosition[2] += right[2] * moveSpeed;
            cameraTarget[0] += right[0] * moveSpeed;
            cameraTarget[2] += right[2] * moveSpeed;
            break;
        case "ArrowUp":
            // Move forward
            cameraPosition[0] += forward[0] * moveSpeed;
            cameraPosition[2] += forward[2] * moveSpeed;
            cameraTarget[0] += forward[0] * moveSpeed;
            cameraTarget[2] += forward[2] * moveSpeed;
            break;
        case "ArrowDown":
            // Move backward
            cameraPosition[0] -= forward[0] * moveSpeed;
            cameraPosition[2] -= forward[2] * moveSpeed;
            cameraTarget[0] -= forward[0] * moveSpeed;
            cameraTarget[2] -= forward[2] * moveSpeed;
            break;
        case "PageUp":
            // Move up
            cameraPosition[1] += moveSpeed;
            cameraTarget[1] += moveSpeed;
            break;
        case "PageDown":
            // Move down
            cameraPosition[1] -= moveSpeed;
            cameraTarget[1] -= moveSpeed;
            break;
    }
});

pointerController.addEventListener("pointermove", (e) => {
    const dx = e.movementX;
    const dy = e.movementY;
    
    const sensitivity = 0.005;
    
    if (e.buttons === 1) {
        // Left button (rotate around target)
        yaw -= dx * sensitivity;
        pitch -= dy * sensitivity;
        
        // Clamp pitch to avoid flipping
        pitch = Math.max(-Math.PI + 0.1, Math.min(-0.1, pitch)); // Looking downward
        
        // Update camera position based on spherical coordinates
        updateCameraOrbit();
    }
});

function updateCameraOrbit() {
    cameraPosition[0] = cameraTarget[0] + distance * Math.cos(pitch) * Math.sin(yaw);
    cameraPosition[1] = cameraTarget[1] + distance * Math.sin(pitch);
    cameraPosition[2] = cameraTarget[2] + distance * Math.cos(pitch) * Math.cos(yaw);
}

// Initialize camera
updateCameraOrbit();

let capturingMatrix = null;
let targetPosition = [
    [0, -0.17, 0],      // Center
//     [0.2, 0, -0.1],    // Right
//     [-0.2, 0, -0.1],   // Left
//     [0, 0.2, -0.1],    // Forward
//     [0, -0.2, -0.1],   // Backward
//     [0.15, 0.15, -0.1], // Diagonal
//     [-0.15, 0.15, -0.1],
//     [0.15, -0.15, -0.1],
//     [0.1, -0.1, -0.1],
//     [-0.1, -0.1, -0.1],
//     [-0.15, -0.15, -0.1]
];
const capturedData = [];
let zip;
document.addEventListener("keydown", async (event) => {
	if (event.key === "T" || event.key === "t") {
		const startTime = performance.now();
		console.log("Capturing the images...");
		zip = new JSZip();

		for (const target of targetPosition) {
			const radius = 0.15; // distance of hemisphere samples
			const poses = camPositionHelper.generateFibonacciHemisphereAroundCamera(
				150,
				radius,
				target
			);
			console.log("Hemisphere poses:", poses);

			let idx = 0;
			for (const pos of poses) {
				idx++;
				if (idx < 62) continue; // For debugging
				if (idx == 63) break; // For debugging
				// Each pose looks at the current camera position
				// Check if camera is too close to target
                const distance = Math.sqrt(
                    Math.pow(pos[0] - target[0], 2) +
                    Math.pow(pos[1] - target[1], 2) +
                    Math.pow(pos[2] - target[2], 2)
                );
                
                if (distance < 0.1) {
                    console.log("Skipping camera too close to target:", distance);
                    continue;
                }

				let {viewMatrix, projectionViewMatrix} = camPositionHelper.computeCameraMatrix(pos, target, canvas);
				capturingMatrix = projectionViewMatrix;
				device.queue.writeBuffer(matrixBuffer, 0, capturingMatrix);

				await render();

				// let { quat, t } = mat4.cameraToColmapPoseEyeTarget(pos, target);
				let { quat, t } = mat4.cameraToColmapPose(pos, viewMatrix);
				let { fx, fy, cx, cy, width, height } = mat4.getIntrinsics(canvas.width, canvas.height, 45 * Math.PI / 180);

				let filename = `image_${capturedData.length + 1}_${target.join("_")}.png`
				capturedData.push({
					imageId: capturedData.length + 1,
					filename: filename,
					quat, t,
					fx, fy, cx, cy, width, height
				});

				composer.initializeCompositeTexture();
				let depthMap = new DepthMap(canvas, device, depthTexture);
				let depthValues = await depthMap.groupDepthIntoBins();
				depthValues.reverse();

				for (let i = 0; i < depthValues.length; i++) {
					console.log(i);
					await renderPointsInDepthRange(
						depthValues[i][0],
						depthValues[i][1],
						// tempBindGroup
					);
				}

				const outputData = await composer.compositeDepths();
				// const outputData = await captureOriginal();
				let blob = await getBlob(outputData, canvas.width, canvas.height);
				zip.file(`images/${filename}`, blob);

				// if (typeof tempMatrixBuffer.destroy === "function") {
				// 	// optional defensive destroy if implemented in environment
				// 	tempMatrixBuffer.destroy();
				// }
			}

		}
		capturingMatrix = null;

		await exportResults(capturedData);

		const endTime = performance.now();
		console.log(`Composite completed in ${endTime - startTime} ms`);
	}
});

async function exportResults(capturedData, points) {
    zip.file("cameras.txt", camPositionHelper.writeCamerasTxt(capturedData));
    zip.file("images.txt", camPositionHelper.writeImagesTxt(capturedData, scaleFactor));
    zip.file("points3D.txt", camPositionHelper.writePoints3DTxt(positions, colorsRGB, scaleFactor));

    // images are already stored under /images
    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, "export_bundle.zip");
}

async function render() {
	const commandEncoder = device.createCommandEncoder();
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

	// Add this after your matrixBuffer creation
	const targetPositionBuffer = device.createBuffer({
		size: 12, // 3 floats (x, y, z)
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	device.queue.writeBuffer(targetPositionBuffer, 0, new Float32Array(targetPosition[0])); // Example target position

	const targetPositionBindGroup = device.createBindGroup({
		layout: renderingPipeline.getBindGroupLayout(3), // Make sure this matches your shader
		entries: [
			{
				binding: 0,
				resource: { buffer: targetPositionBuffer },
			},
		],
	});

	renderPass.setPipeline(renderingPipeline);
	renderPass.setBindGroup(1, matrixBindGroup);
	renderPass.setBindGroup(2, depthRangeBindGroup); // Apply depth range filter
	renderPass.setBindGroup(3, targetPositionBindGroup); // Set target position

	for (const pointcloud of pointclouds) {
		renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
		renderPass.draw(pointcloud.numberOfPoints);
	}
	renderPass.end();

	device.queue.submit([commandEncoder.finish()]);
}

document.addEventListener("keydown", async (event) => {
	if (event.key === "E" || event.key === "e") {
		const startTime = performance.now();

		// let orig_images = [
		// 	"../data/images/image3.png",
		// ];
		// let rec_images = [
		// 	"../data/images/image3_50_alpha.png",
		// 	"../data/images/image3_80_alpha.png",
		// 	"../data/images/image3_90_alpha.png",
		// 	"../data/images/image3_95_alpha.png",
		// 	"../data/images/image3_99_alpha.png",
		// ];

		let orig_images = [
			"../data/images/image3.png",
		];
		let rec_images = [
			"../data/images/image4.png",
			"../data/images/image4_50_alpha.png",
			"../data/images/image4_90_alpha.png",
			"../data/images/image4_80_alpha.png",
			"../data/images/image4_95_alpha.png",
			"../data/images/image4_99_alpha.png",
		];
		let maxIterations = [2000];

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
		// 			orig_images[0],
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
		// 			orig_images[0],
		// 			rec_images[j]
		// 		);
		// 		let psnr = await evaluator.evaluate_sor();
		// 		console.log("PSNR = ", psnr);
		// 	}
		// }

		let nSoves = [20];
		let nSmooths = [50];
		// for (let j = 0; j < rec_images.length; j++) {
		// 	console.log("IMAGE ", rec_images[j]);
		// 	for (let i = 0; i < nSoves.length; i++) {
		// 		console.log(
		// 			"Reconstruction evaluation for ",
		// 			nSoves[i],
		// 			" nSolves"
		// 		);
		// 		for (let k = 0; k < nSmooths.length; k++) {
		// 			console.log(
		// 				"Reconstruction evaluation for ",
		// 				nSmooths[k],
		// 				" nSmooth"
		// 			);
		// 			const evaluator = new Evaluation(
		// 				device,
		// 				canvas,
		// 				canvas.width,
		// 				canvas.height,
		// 				0,
		// 				nSoves[i],
		// 				nSmooths[k],
		// 				orig_images[0],
		// 				rec_images[j]
		// 			);
		// 			let psnr = await evaluator.evaluate_multigrid();
		// 			console.log("PSNR = ", psnr);
		// 		}
		// 	}
		// }

		const evaluator = new Evaluation(
			device,
			canvas,
			canvas.width,
			canvas.height,
			10000,
			nSoves[0],
			nSmooths[0],
			orig_images[0],
			rec_images[4]
		);
		// let psnr = await evaluator.evaluate_jacobi();
		// let psnr2 = await evaluator.evaluate_sor();
		await evaluator.evaluate_multigrid();
	}
});

async function captureOriginal(myMatrixBindingGroup = matrixBindGroup) {
	// Create a buffer for depth range
	let depthRangeBuffer = device.createBuffer({
		size: 8, // Two floats (4 bytes each)
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	// Write the min and max depth to the buffer
	device.queue.writeBuffer(
		depthRangeBuffer,
		0,
		new Float32Array([0, 1])
	);

	// Create a bind group for depth filtering
	let depthRangeBindGroup = device.createBindGroup({
		layout: renderingPipeline.getBindGroupLayout(2),
		entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
	});

// Add this after your matrixBuffer creation
const targetPositionBuffer = device.createBuffer({
    size: 12, // 3 floats (x, y, z)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

	device.queue.writeBuffer(targetPositionBuffer, 0, Float32Array.from(targetPosition[0])); // Example target position

const targetPositionBindGroup = device.createBindGroup({
    layout: renderingPipeline.getBindGroupLayout(3), // Make sure this matches your shader
    entries: [
        {
            binding: 0,
            resource: { buffer: targetPositionBuffer },
        },
    ],
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
	renderPass.setBindGroup(1, myMatrixBindingGroup);
	renderPass.setBindGroup(2, depthRangeBindGroup); // Apply depth range filter
	renderPass.setBindGroup(3, targetPositionBindGroup); // Set target position

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
	return imageData
}

async function renderPointsInDepthRange(minDepth, maxDepth, myMatrixBindingGroup = matrixBindGroup) {
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

	// Add this after your matrixBuffer creation
	const targetPositionBuffer = device.createBuffer({
		size: 12, // 3 floats (x, y, z)
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	device.queue.writeBuffer(targetPositionBuffer, 0, Float32Array.from(targetPosition[0])); // Example target position

	const targetPositionBindGroup = device.createBindGroup({
		layout: renderingPipeline.getBindGroupLayout(3), // Make sure this matches your shader
		entries: [
			{
				binding: 0,
				resource: { buffer: targetPositionBuffer },
			},
		],
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
	renderPass.setBindGroup(1, myMatrixBindingGroup);
	renderPass.setBindGroup(2, depthRangeBindGroup); // Apply depth range filter
	renderPass.setBindGroup(3, targetPositionBindGroup); // Set target position
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

	let solver = new Solver(canvas, device);
	let image = await solver.sorRedBlack(
		captureTexture,
		reconstructionRead,
		reconstructionWrite
	);

	// let multigridSolver = new MultigridSolver(canvas, device);
	// let image = await multigridSolver.multigridSolve(captureTexture);

	// let cgSolver = new ConjugateGradientSolver(canvas, device);
	// let image = await cgSolver.conjGradientSolve(captureTexture, reconstructionRead, reconstructionWrite);

	await saveTextureToPNG(
		image,
		canvas.width,
		canvas.height,
		`depth_layer_${minDepth}_${maxDepth}.png`
	);

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
	// const modelMatrix = mat4.multiply(
	// 	mat4.rotateX(verticalAngle), // Rotate around X-axis (up-down)
	// 	mat4.rotateZ(horizontalAngle) // Rotate around Z-axis (left-right)
	// );

	// Apply the position adjustments from the right-click drag (left-right, up-down)
	// const viewMatrix = mat4.translation([moveX, moveY, -zoomFactor]);

	// const projectionMatrix = mat4.perspective(
	// 	1,
	// 	canvas.width / canvas.height,
	// 	0.1,
	// 	100
	// );
	// if (capturingMatrix == null) {
	// 	const matrix = mat4.multiply(projectionMatrix, viewMatrix, modelMatrix);

	// 	device.queue.writeBuffer(matrixBuffer, 0, matrix);
	// } else {
	// 	device.queue.writeBuffer(matrixBuffer, 0, capturingMatrix);
	// }

	// cameraPosition[0] = cameraTarget[0] + zoomFactor * Math.cos(verticalAngle) * Math.sin(horizontalAngle);
    // cameraPosition[1] = cameraTarget[1] + zoomFactor * Math.sin(verticalAngle);
    // cameraPosition[2] = cameraTarget[2] + zoomFactor * Math.cos(verticalAngle) * Math.cos(horizontalAngle);
    
	// console.log("Camera position:", cameraPosition, " Target:", cameraTarget);
//     const viewMatrix = mat4.lookAt(cameraPosition, cameraTarget, cameraUp);
//     const projectionMatrix = mat4.perspective(
//         1,
//         canvas.width / canvas.height,
//         0.0001,
//         10000
//     );
    
//     const matrix = mat4.multiply(projectionMatrix, viewMatrix);
    
//     if (capturingMatrix == null) {
//         device.queue.writeBuffer(matrixBuffer, 0, matrix);
//     } else {
//         device.queue.writeBuffer(matrixBuffer, 0, capturingMatrix);
//     }
	

// 	const renderPass = commandEncoder.beginRenderPass({
// 		colorAttachments: [
// 			{
// 				view: context.getCurrentTexture().createView(),
// 				loadOp: "clear",
// 				clearValue: [0, 0, 0, 0],
// 				storeOp: "store",
// 			},
// 		],
// 		depthStencilAttachment: {
// 			view: depthTexture.createView(),
// 			depthLoadOp: "clear",
// 			depthClearValue: 1,
// 			depthStoreOp: "store",
// 		},
// 	});

// 	// Create a buffer for depth range
// 	let depthRangeBuffer = device.createBuffer({
// 		size: 8, // Two floats (4 bytes each)
// 		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
// 	});

// 	// Write the min and max depth to the buffer
// 	device.queue.writeBuffer(depthRangeBuffer, 0, new Float32Array([0, 1]));

// 	// Create a bind group for depth filtering
// 	let depthRangeBindGroup = device.createBindGroup({
// 		layout: renderingPipeline.getBindGroupLayout(2),
// 		entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
// 	});

// 	// Add this after your matrixBuffer creation
// const targetPositionBuffer = device.createBuffer({
//     size: 12, // 3 floats (x, y, z)
//     usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
// });

// 	device.queue.writeBuffer(targetPositionBuffer, 0, new Float32Array(targetPosition[0])); // Example target position

// const targetPositionBindGroup = device.createBindGroup({
//     layout: renderingPipeline.getBindGroupLayout(3), // Make sure this matches your shader
//     entries: [
//         {
//             binding: 0,
//             resource: { buffer: targetPositionBuffer },
//         },
//     ],
// });

// 	renderPass.setPipeline(renderingPipeline);
// 	renderPass.setBindGroup(1, matrixBindGroup);
// 	renderPass.setBindGroup(2, depthRangeBindGroup); // Apply depth range filter
// 	renderPass.setBindGroup(3, targetPositionBindGroup); // Set target position

// 	for (const pointcloud of pointclouds) {
// 		renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
// 		renderPass.draw(pointcloud.numberOfPoints);
// 	}
// 	renderPass.end();

	device.queue.submit([commandEncoder.finish()]);

	requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
