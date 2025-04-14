// Updated main.js with position padding corrections
import * as mat4 from "./js/mat4.js";
import { PointerController } from "./js/PointerController.js";
import { saveTextureToPNG } from "./js/Utils.js";
import { Solver } from "./js/Solvers.js";
import { DepthMap } from "./js/DepthMap.js";
import { Composer } from "./js/Composer.js";
import { LasLoader } from "./js/LasLoader.js";

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

const composer = new Composer(document, canvas, device, format);
composer.initializeBackground("./data/sky.jpg")

const randomizationCode = await fetch("./shaders/randomization.wgsl").then((response) =>
	response.text()
);
const renderingCode = await fetch("./shaders/rendering.wgsl").then((response) =>
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

const jacobiaCode = await fetch("./shaders/jacobia.wgsl").then((response) =>
	response.text()
);

const updateRedBlackWGSL = await fetch("./shaders/updateRedBlack.wgsl").then((response) =>
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

const lasLoader = new LasLoader(lasFile);
const lasData = await lasLoader.loadLasData();
const positions = lasData.positions;
const colors = lasData.colors;

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

	// readParticleBuffer(stagingBuffer);
}

// Used for debugging
async function readParticleBuffer(stagingBuffer) {
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

document.addEventListener("keydown", async (event) => {
	if (event.key === "T" || event.key === "t") {
		console.log("Capturing the current view...");

		composer.initializeCompositeTexture();

		let depthMap = new DepthMap(canvas, device, depthTexture);
		let depthValues = await depthMap.groupDepthIntoBins();
		depthValues.reverse();

		for (let i = 0; i < depthValues.length; i++) {
			await renderPointsInDepthRange(
				depthValues[i][0],
				depthValues[i][1]
			);
		}

		const outputBuffer = device.createBuffer({
            size: canvas.width * canvas.height * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

		const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            { texture: composer.compositeResult },
            {
                buffer: outputBuffer,
                bytesPerRow: canvas.width * 4,
                rowsPerImage: canvas.height
            },
            [canvas.width, canvas.height, 1]
        );
        device.queue.submit([commandEncoder.finish()]);
        
        await outputBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = outputBuffer.getMappedRange();
        const imageData = new Uint8Array(arrayBuffer);
        
        // Save the final composite
        saveTextureToPNG(imageData, canvas.width, canvas.height, "composite_final.png");
        
        outputBuffer.unmap();
        outputBuffer.destroy();
        composer.compositeResult.destroy();
        composer.compositeResult = null;
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

	saveTextureToPNG(image, canvas.width, canvas.height, `depth_layer_${minDepth}_${maxDepth}.png`);

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

	for (const particleSystem of particleSystems) {
		renderPass.setBindGroup(0, particleSystem.renderingBindGroup);
		renderPass.draw(particleSystem.numberOfParticles);
	}
	renderPass.end();

	device.queue.submit([commandEncoder.finish()]);

	requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
