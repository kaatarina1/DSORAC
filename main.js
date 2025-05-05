// Updated main.js with position padding corrections
import * as mat4 from "./js/mat4.js";
import { PointerController } from "./js/PointerController.js";
import { saveTextureToPNG } from "./js/Utils.js";
import { Solver } from "./js/Solvers.js";
import { DepthMap } from "./js/DepthMap.js";
import { Composer } from "./js/Composer.js";
import { LasLoader } from "./js/LasLoader.js";
import { MultigridSolver } from "./js/MultigridSolver.js";

const adapter = await navigator.gpu.requestAdapter();
const hasBGRA8unormStorage = adapter.features.has("bgra8unorm-storage");
const device = await adapter?.requestDevice({
	requiredFeatures: hasBGRA8unormStorage ? ["bgra8unorm-storage"] : [],
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
        pointDataView.setFloat32(pointOffset + 4, positions[posIndex + 1], true);
        pointDataView.setFloat32(pointOffset + 8, positions[posIndex + 2], true);
        
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

let horizontalAngle = 0; // Angle for horizontal rotation (around Z-axis)
let verticalAngle = 0; // Angle for vertical rotation (around X-axis)
let moveX = 0; // Left-right movement
let moveY = 0; // Up-down movement
let zoomFactor = 1.2; // Default zoom factor

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
		depthValues.reverse();

		for (let i = 0; i < depthValues.length; i++) {
			console.log(i);
			await renderPointsInDepthRange(
				depthValues[i][0],
				depthValues[i][1]
			);
		}

		const outputBuffer = device.createBuffer({
			size: canvas.width * canvas.height * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		const commandEncoder = device.createCommandEncoder();
		commandEncoder.copyTextureToBuffer(
			{ texture: composer.compositeResult },
			{
				buffer: outputBuffer,
				bytesPerRow: canvas.width * 4,
				rowsPerImage: canvas.height,
			},
			[canvas.width, canvas.height, 1]
		);
		device.queue.submit([commandEncoder.finish()]);

		await outputBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = outputBuffer.getMappedRange();
		const imageData = new Uint8Array(arrayBuffer);

		// Save the final composite
		await saveTextureToPNG(
			imageData,
			canvas.width,
			canvas.height,
			"composite_final.png"
		);
		outputBuffer.unmap();
		outputBuffer.destroy();
		composer.compositeResult.destroy();
		composer.compositeResult = null;
		const endTime = performance.now();
		console.log(`Composite completed in ${endTime - startTime} ms`);
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
		format: format,
		usage:
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
	});

	commandEncoder.copyTextureToTexture(
		{
			texture: captureTexture,
			mipLevel: 0,
			origin: { x: 0, y: 0, z: 0 }, // Start copying from the top-left corner
		},
		{
			texture: reconstructionRead,
			mipLevel: 0,
			origin: { x: 0, y: 0, z: 0 }, // Copy to the same top-left corner
		},
		[canvas.width, canvas.height, 1] // Copy the entire texture
	);

	let reconstructionWrite = device.createTexture({
		size: [canvas.width, canvas.height],
		format: format,
		usage:
			GPUTextureUsage.STORAGE_BINDING |
			GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
	});
	device.queue.submit([commandEncoder.finish()]);

	// let solver = new Solver(canvas, device);
	// let image = await solver.sorWithResidual(
	// 	captureTexture,
	// 	reconstructionRead,
	// 	reconstructionWrite
	// );

	let multigridSolver = new MultigridSolver(canvas, device);
	let image = await multigridSolver.multigridSolve(captureTexture);

	// Read the buffer
	// await outputBuffer.mapAsync(GPUMapMode.READ);
	// const arrayBuffer = outputBuffer.getMappedRange();
	// const imageData = new Uint8Array(arrayBuffer);

	await saveTextureToPNG(
		image,
		canvas.width,
		canvas.height,
		`depth_layer_${minDepth}_${maxDepth}.png`
	);

	await composer.compositeLayer(image);

	captureTexture.destroy();
	reconstructionRead.destroy();
	reconstructionWrite.destroy();
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
