// Updated main.js with position padding corrections
import * as mat4 from "./mat4.js";
import { PointerController } from "./pointerController.js";
import { createLazPerf } from "laz-perf";
import { parseHeader, saveTextureToPNG } from "./utils.js";
import { Solver } from "./Solvers.js";
import { DepthMap } from "./DepthMap.js";

const adapter = await navigator.gpu.requestAdapter();
const hasBGRA8unormStorage = adapter.features.has("bgra8unorm-storage");
const device = await adapter?.requestDevice({
	requiredFeatures: hasBGRA8unormStorage ? ["bgra8unorm-storage"] : [],
});
const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");
const format = hasBGRA8unormStorage
	? navigator.gpu.getPreferredCanvasFormat()
	: "rgba8unorm"; // format za shranjevanje slik rgb 32 float, -> slike shranjevat v pam
// const imageFormat = "rgba32float";
const alphaMode = "premultiplied";
context.configure({ device, format, alphaMode: "premultiplied" });

const randomizationCode = await fetch("randomization.wgsl").then((response) =>
	response.text()
);
const renderingCode = await fetch("rendering.wgsl").then((response) =>
	response.text()
);

const randomizationModule = device.createShaderModule({
	code: randomizationCode,
});
const renderingModule = device.createShaderModule({ code: renderingCode });

const randomizationPipeline = device.createComputePipeline({
	compute: {
		module: randomizationModule,
	},
	layout: "auto",
});

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

// const floatRenderingPipeline = device.createRenderPipeline({
// 	vertex: {
// 		module: renderingModule,
// 	},
// 	fragment: {
// 		module: renderingModule,
// 		targets: [{ format: imageFormat }],
// 	},
// 	primitive: {
// 		topology: "point-list",
// 	},
// 	depthStencil: {
// 		depthWriteEnabled: true,
// 		depthCompare: "less",
// 		format: "depth32float",
// 	},
// 	layout: "auto",
// });

const jacobiaCode = await fetch("jacobia.wgsl").then((response) =>
	response.text()
);

const updateRedBlackWGSL = await fetch("updateRedBlack.wgsl").then((response) =>
	response.text()
);

const jacobiaModule = device.createShaderModule({
	code: jacobiaCode,
});

const jacobiaPipeline = device.createComputePipeline({
	compute: {
		module: jacobiaModule,
		entryPoint: "main",
	},
	layout: "auto",
});

const updateRedBlackModule = device.createShaderModule({
	code: updateRedBlackWGSL,
});

const updateRedBlackPipeline = device.createComputePipeline({
	layout: "auto",
	compute: {
		module: updateRedBlackModule,
		entryPoint: "main",
	},
});

const lasFile = `./data/cropped_filtered_1.las`;

const LazPerf = await createLazPerf({
	locateFile: (file) => `./node_modules/laz-perf/lib/laz-perf.wasm`,
});
const response = await fetch(lasFile);
const arrayBuffer = await response.arrayBuffer();
const file = new Uint8Array(arrayBuffer);

const header = parseHeader(file);
const {
	pointDataRecordFormat,
	pointDataRecordLength,
	pointCount,
	scale,
	offset,
} = header;

console.log("Header Info:", header);

const laszip = new LazPerf.LASZip();
const dataPtr = LazPerf._malloc(pointDataRecordLength);
const filePtr = LazPerf._malloc(file.byteLength);

LazPerf.HEAPU8.set(
	new Uint8Array(file.buffer, file.byteOffset, file.byteLength),
	filePtr
);

laszip.open(filePtr, file.byteLength);

const positions = new Float32Array(laszip.getCount() * 3);
const colors = new Uint32Array(laszip.getCount());

let minX = Infinity,
	minY = Infinity,
	minZ = Infinity;
let maxX = -Infinity,
	maxY = -Infinity,
	maxZ = -Infinity;

for (let i = 0; i < laszip.getCount(); ++i) {
	laszip.getPoint(dataPtr);

	const pointBuffer = new DataView(
		LazPerf.HEAPU8.buffer,
		dataPtr,
		pointDataRecordLength
	);

	const x = pointBuffer.getInt32(0, true) * scale[0] + offset[0];
	const y = pointBuffer.getInt32(4, true) * scale[1] + offset[1];
	const z = pointBuffer.getInt32(8, true) * scale[2] + offset[2];

	// console.log(`Point ${i}: x = ${x}, y = ${y}, z = ${z}`);

	const r = pointBuffer.getUint16(28, true) / 65535;
	const g = pointBuffer.getUint16(30, true) / 65535;
	const b = pointBuffer.getUint16(32, true) / 65535;

	minX = Math.min(minX, x);
	maxX = Math.max(maxX, x);
	minY = Math.min(minY, y);
	maxY = Math.max(maxY, y);
	minZ = Math.min(minZ, z);
	maxZ = Math.max(maxZ, z);

	positions.set([x, y, z], i * 3);
	const packedColor =
		((Math.round(r * 255) & 0xff) << 0) | // Red channel in the least significant byte
		((Math.round(g * 255) & 0xff) << 8) | // Green channel in the second byte
		((Math.round(b * 255) & 0xff) << 16) | // Blue channel in the third byte
		(0xff << 24); // Alpha channel in the most significant byte
	const unsignedPackedColor = packedColor >>> 0; // Force unsigned 32-bit representation
	// if (i > 10000 && i < 10010) {
	// 	console.log(
	// 		"R: ",
	// 		Math.round(r * 255),
	// 		"G: ",
	// 		Math.round(g * 255),
	// 		"B: ",
	// 		Math.round(b * 255)
	// 	);
	// 	console.log(`Packed Color: 0x${unsignedPackedColor.toString(16)}`);
	// }
	colors.set([unsignedPackedColor], i);
}

const centerX = (minX + maxX) / 2;
const centerY = (minY + maxY) / 2;
const centerZ = (minZ + maxZ) / 2;
const scaleFactor = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

// Normalize positions to fit [-1, 1] in all axes
for (let i = 0; i < positions.length; i += 3) {
	positions[i] = (positions[i] - centerX) / scaleFactor;
	positions[i + 1] = (positions[i + 1] - centerY) / scaleFactor;
	positions[i + 2] = (positions[i + 2] - centerZ) / scaleFactor;
}
LazPerf._free(filePtr);
LazPerf._free(dataPtr);
laszip.delete();

const numberOfAllParticles = positions.length / 3;
const maxNumberOfParticlesPerBuffer = 1024 * 1024;
const particleByteSize = 16;

console.log(numberOfAllParticles);
console.log(numberOfAllParticles / maxNumberOfParticlesPerBuffer);

function createParticleSystem(numberOfParticles, count) {
	if (numberOfParticles > maxNumberOfParticlesPerBuffer) {
		throw new Error("Too many particles for one particle buffer");
	}

	const paddedPositions = new Float32Array(numberOfParticles * 4);
	const splitColors = new Uint32Array(numberOfParticles);
	let index = 0;
	let colorIndex = count * numberOfParticles;
	for (
		let i = count * numberOfParticles;
		i < (count + 1) * numberOfParticles;
		i++
	) {
		paddedPositions[index * 4 + 0] = positions[i * 3 + 0];
		paddedPositions[index * 4 + 1] = positions[i * 3 + 1];
		paddedPositions[index * 4 + 2] = positions[i * 3 + 2];
		paddedPositions[index * 4 + 3] = 0.0; // Padding
		splitColors[index] = colors[colorIndex];
		index++;
		colorIndex++;
	}

	const particleBuffer = device.createBuffer({
		size: numberOfParticles * particleByteSize,
		usage:
			GPUBufferUsage.STORAGE |
			GPUBufferUsage.COPY_DST |
			GPUBufferUsage.COPY_SRC,
	});

	const uniformBuffer = device.createBuffer({
		size: 4,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	device.queue.writeBuffer(
		uniformBuffer,
		0,
		new Uint32Array([numberOfParticles])
	);

	// Create GPU buffers for positions and colors
	const positionsBuffer = device.createBuffer({
		size: paddedPositions.byteLength,
		usage:
			GPUBufferUsage.STORAGE |
			GPUBufferUsage.COPY_DST |
			GPUBufferUsage.COPY_SRC,
	});
	device.queue.writeBuffer(positionsBuffer, 0, paddedPositions);

	const colorsBuffer = device.createBuffer({
		size: splitColors.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(colorsBuffer, 0, splitColors);

	const randomizationBindGroup = device.createBindGroup({
		layout: randomizationPipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: positionsBuffer } },
			{ binding: 1, resource: { buffer: colorsBuffer } },
			{ binding: 2, resource: { buffer: particleBuffer } },
			{ binding: 3, resource: { buffer: uniformBuffer } },
		],
	});

	const renderingBindGroup = device.createBindGroup({
		layout: renderingPipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: { buffer: particleBuffer },
			},
		],
	});

	return {
		particleBuffer,
		uniformBuffer,
		positionsBuffer,
		colorsBuffer,
		randomizationBindGroup,
		renderingBindGroup,
		numberOfParticles,
	};
}

function randomizeParticles(particleSystem) {
	const commandEncoder = device.createCommandEncoder();

	const computePass = commandEncoder.beginComputePass();
	computePass.setPipeline(randomizationPipeline);
	computePass.setBindGroup(0, particleSystem.randomizationBindGroup);
	const numberOfWorkgroups = Math.ceil(
		particleSystem.numberOfParticles / 256
	);
	computePass.dispatchWorkgroups(numberOfWorkgroups);

	computePass.end();

	device.queue.submit([commandEncoder.finish()]);
}

const particleSystems = [];
for (let i = 0; i < numberOfAllParticles / maxNumberOfParticlesPerBuffer; i++) {
	const particleSystem = createParticleSystem(
		numberOfAllParticles > maxNumberOfParticlesPerBuffer
			? maxNumberOfParticlesPerBuffer
			: numberOfAllParticles,
		i
	);
	randomizeParticles(particleSystem);
	particleSystems.push(particleSystem);

	const stagingBuffer = device.createBuffer({
		size: particleSystem.numberOfParticles * particleByteSize,
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});
	const commandEncoder = device.createCommandEncoder();
	commandEncoder.copyBufferToBuffer(
		particleSystem.particleBuffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		particleSystem.numberOfParticles * particleByteSize // Size of data to copy
	);
	device.queue.submit([commandEncoder.finish()]);

	// readParticleBuffer(
	// 	particleSystem.particleBuffer,
	// 	stagingBuffer,
	// 	particleSystem.numberOfParticles
	// );
}

async function readParticleBuffer(
	particleBuffer,
	stagingBuffer,
	numberOfParticles
) {
	// Ensure the buffer is mapped for reading
	await stagingBuffer.mapAsync(GPUMapMode.READ);

	// Access the buffer data
	const arrayBuffer = stagingBuffer.getMappedRange();

	// Interpret the buffer data
	const particleData = new DataView(arrayBuffer);

	for (let i = 0; i < 3; i++) {
		const baseOffset = i * particleByteSize; // Assuming each particle is 24 bytes
		const x = particleData.getFloat32(baseOffset, true); // Read position.x
		const y = particleData.getFloat32(baseOffset + 4, true); // Read position.y
		const z = particleData.getFloat32(baseOffset + 8, true); // Read position.z
		const color = particleData.getUint32(baseOffset + 12, true); // Read packed color

		console.log(
			`Particle ${i}: Position = (${x}, ${y}, ${z}), Color = 0x${color.toString(
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

function filterImage(image, data) {
	// Create a copy of the data to store filtered pixels
	const filteredData = new Uint8Array(image);

	// Define the kernel size for the filter
	const kernelSize = 3; // 3x3 kernel
	const halfKernel = Math.floor(kernelSize / 2);

	// Function to get pixel indices in the 1D array
	const getPixelIndex = (x, y) => y * x * 4;

	// Loop over each pixel in the image
	for (let y = halfKernel; y < canvas.height - halfKernel; y++) {
		for (let x = halfKernel; x < canvas.width - halfKernel; x++) {
			const currentPixelIndex = getPixelIndex(x, y);

			// Skip pixels with alpha == 0
			if (image[currentPixelIndex + 3] === 0) continue;

			const neighbors = { r: [], g: [], b: [] };

			for (let ky = -halfKernel; ky <= halfKernel; ky++) {
				for (let kx = -halfKernel; kx <= halfKernel; kx++) {
					const neighborIndex = getPixelIndex(x + kx, y + ky);

					// Only consider neighbors with alpha > 0
					if (image[neighborIndex + 3] !== 0) {
						neighbors.r.push(image[neighborIndex]);
						neighbors.g.push(image[neighborIndex + 1]);
						neighbors.b.push(image[neighborIndex + 2]);
					}
				}
			}

			// Skip pixels with no valid neighbors
			if (neighbors.r.length === 0) continue;

			neighbors.r.sort((a, b) => a - b);
			neighbors.g.sort((a, b) => a - b);
			neighbors.b.sort((a, b) => a - b);

			const medianIndex = Math.floor(neighbors.r.length / 2);

			filteredData[currentPixelIndex] = neighbors.r[medianIndex]; // Red
			filteredData[currentPixelIndex + 1] = neighbors.g[medianIndex]; // Green
			filteredData[currentPixelIndex + 2] = neighbors.b[medianIndex]; // Blue
			filteredData[currentPixelIndex + 3] = image[currentPixelIndex + 3]; // Alpha (unchanged)
		}
	}

	return filteredData;
}

document.addEventListener("keydown", async (event) => {
	if (event.key === "T" || event.key === "t") {
		console.log("Capturing the current view...");

		// Create a texture to store the captured frame
		const captureTexture = device.createTexture({
			size: [canvas.width, canvas.height],
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.RENDER_ATTACHMENT,
			format: format,
		});

		// Configure a render pass for capture
		const commandEncoder = device.createCommandEncoder();
		const renderPass = commandEncoder.beginRenderPass({
			colorAttachments: [
				{
					view: captureTexture.createView(),
					loadOp: "clear",
					clearValue: [0, 0, 0, 0], // Black background
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthTexture.createView(),
				depthLoadOp: "clear",
				depthClearValue: 1,
				depthStoreOp: "discard",
			},
		});

		// Render the current frame into the capture texture
		renderPass.setPipeline(renderingPipeline);
		renderPass.setBindGroup(1, matrixBindGroup);
		for (const particleSystem of particleSystems) {
			renderPass.setBindGroup(0, particleSystem.renderingBindGroup);
			renderPass.draw(particleSystem.numberOfParticles);
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

		// Read the buffer
		await outputBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = outputBuffer.getMappedRange();
		const imageData = new Uint8Array(arrayBuffer);

		// const filteredImage = filterImage(imageData, arrayBuffer);

		saveTextureToPNG(imageData, canvas.width, canvas.height);

		let reconstructionWrite = device.createTexture({
			size: [canvas.width, canvas.height],
			format: format,
			usage:
				GPUTextureUsage.STORAGE_BINDING |
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC,
		});

		// let solver = new Solver(canvas, device, jacobiaPipeline);
		// let image = await solver.jacobian(captureTexture, reconstructionRead, reconstructionWrite);
		// let solver = new Solver(canvas, device, updateRedBlackPipeline);
		// let image = await solver.sorRedBlack(captureTexture, reconstructionRead, reconstructionWrite);
		// saveTextureToPNG(image, canvas.width, canvas.height);

		let depthMap = new DepthMap(canvas, device, depthTexture);
		let depthValues = await depthMap.groupDepthIntoBins();

		for (let i = 0; i < depthValues.length; i++) {
			await renderPointsInDepthRange(
				depthValues[i][0],
				depthValues[i][1]
			);
		}

		// Submit the commands
		// Cleanup resources
		outputBuffer.unmap();
		captureTexture.destroy();
		reconstructionRead.destroy();
		reconstructionWrite.destroy();
		outputBuffer.destroy();
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

	for (const particleSystem of particleSystems) {
		renderPass.setBindGroup(0, particleSystem.renderingBindGroup);
		renderPass.draw(particleSystem.numberOfParticles);
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

	let solver = new Solver(canvas, device, jacobiaPipeline);
	let image = await solver.jacobian(
		captureTexture,
		reconstructionRead,
		reconstructionWrite
	);

	// Read the buffer
	// await outputBuffer.mapAsync(GPUMapMode.READ);
	// const arrayBuffer = outputBuffer.getMappedRange();
	// const imageData = new Uint8Array(arrayBuffer);

	saveTextureToPNG(image, canvas.width, canvas.height);

	outputBuffer.unmap();
	captureTexture.destroy();
	reconstructionRead.destroy();
	reconstructionWrite.destroy();
	outputBuffer.destroy();
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

	for (const particleSystem of particleSystems) {
		renderPass.setBindGroup(0, particleSystem.renderingBindGroup);
		renderPass.draw(particleSystem.numberOfParticles);
	}
	renderPass.end();

	device.queue.submit([commandEncoder.finish()]);

	requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
