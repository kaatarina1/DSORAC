import { Buffer } from "buffer";
import { loadShader } from "./LoadShader.js";

export function parseHeader(fileInput) {
	const file = Buffer.from(fileInput);

	if (file.byteLength < 227) throw new Error("Invalid file length");

	const pointDataRecordFormat = file.readUint8(104) & 0b1111;
	const pointDataRecordLength = file.readUint16LE(105);
	const pointDataOffset = file.readUint32LE(96);
	const pointCount = file.readUint32LE(107);

	const scale = [
		file.readDoubleLE(131),
		file.readDoubleLE(139),
		file.readDoubleLE(147),
	];
	const offset = [
		file.readDoubleLE(155),
		file.readDoubleLE(163),
		file.readDoubleLE(171),
	];
	const min = [
		file.readDoubleLE(187),
		file.readDoubleLE(203),
		file.readDoubleLE(219),
	];
	const max = [
		file.readDoubleLE(179),
		file.readDoubleLE(195),
		file.readDoubleLE(211),
	];
	return {
		pointDataRecordFormat,
		pointDataRecordLength,
		pointDataOffset,
		pointCount,
		scale,
		offset,
		min,
		max,
	};
}

/**
 * Brskalnik zavrne del prenosov, če jih sprožimo v hitrem zaporedju (in
 * URL.revokeObjectURL takoj po click() lahko prenos prekine), zato prenose
 * serializiramo in objektni URL sprostimo šele po zamiku.
 */
let downloadQueue = Promise.resolve();

export function queueBlobDownload(blob, fileName) {
	downloadQueue = downloadQueue.then(async () => {
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = fileName;
		a.style.display = "none";
		document.body.appendChild(a);
		a.click();
		await new Promise((resolve) => setTimeout(resolve, 300));
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	});
	return downloadQueue;
}

/**
 * Shrani teksturo kot PNG. Dela tudi v workerju (OffscreenCanvas + postMessage).
 *
 * `scale` pretvori vrednosti v 0..255: rgba32float podatki so tipično v 0..1,
 * zato se privzeto pomnožijo z 255, Uint8 podatki pa ostanejo nespremenjeni.
 */
export async function saveTextureToPNG(
	imageData,
	width,
	height,
	fileName,
	{ scale = imageData instanceof Float32Array ? 255 : 1 } = {}
) {
	// V workerju ni document — uporabimo OffscreenCanvas
	const inWorker = typeof document === "undefined";
	const canvas = inWorker
		? new OffscreenCanvas(width, height)
		: document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d");
	const imageDataObj = ctx.createImageData(width, height);

	// Copy the pixel data to the ImageData object
	for (let i = 0; i < imageData.length; i += 4) {
		// BGRA to RGBA
		imageDataObj.data[i] = imageData[i] * scale;
		imageDataObj.data[i + 1] = imageData[i + 1] * scale;
		imageDataObj.data[i + 2] = imageData[i + 2] * scale;
		imageDataObj.data[i + 3] = imageData[i + 3] * scale;
	}
	ctx.putImageData(imageDataObj, 0, 0);

	if (inWorker) {
		// Worker ne more sprožiti prenosa — blob pošljemo glavni niti
		const blob = await canvas.convertToBlob({ type: "image/png" });
		self.postMessage({ type: "SAVE_PNG", blob, fileName });
		return;
	}

	// Step 5: Convert canvas content to PNG and trigger download
	const blob = await new Promise((resolve) =>
		canvas.toBlob(resolve, "image/png")
	);
	await queueBlobDownload(blob, fileName);
}

export async function saveNormalizedTextureToPNG(imageData, width, height, fileName) {
	// V workerju ni document — uporabimo OffscreenCanvas
	const inWorker = typeof document === "undefined";
	const canvas = inWorker
		? new OffscreenCanvas(width, height)
		: document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d");
	const imageDataObj = ctx.createImageData(width, height);

	// Copy the pixel data to the ImageData object
	for (let i = 0; i < imageData.length; i += 4) {
		// BGRA to RGBA
		imageDataObj.data[i] = imageData[i] * 180;
		imageDataObj.data[i + 1] = imageData[i + 1] * 100;
		imageDataObj.data[i + 2] = imageData[i + 2] * 200;
		imageDataObj.data[i + 3] = imageData[i + 3] * 255;
	}
	ctx.putImageData(imageDataObj, 0, 0);

	if (inWorker) {
		// Worker ne more sprožiti prenosa — blob pošljemo glavni niti
		const blob = await canvas.convertToBlob({ type: "image/png" });
		self.postMessage({ type: "SAVE_PNG", blob, fileName });
		return;
	}

	// Step 5: Convert canvas content to PNG and trigger download
	const dataURL = canvas.toDataURL("image/png");
	const a = document.createElement("a");
	a.href = dataURL;
	a.download = fileName;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
}

export async function getBlob(imageData, width, height) {
	// Create an offscreen canvas
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imageDataObj = ctx.createImageData(width, height);

    // Copy pixel data (BGRA → RGBA if needed)
    for (let i = 0; i < imageData.length; i += 4) {
        imageDataObj.data[i]     = imageData[i];     // R
        imageDataObj.data[i + 1] = imageData[i + 1]; // G
        imageDataObj.data[i + 2] = imageData[i + 2]; // B
        imageDataObj.data[i + 3] = imageData[i + 3]; // A
    }
    ctx.putImageData(imageDataObj, 0, 0);

    // Return PNG as a Blob instead of downloading
    return new Promise((resolve) => {
        canvas.toBlob((blob) => {
            resolve(blob);
        }, "image/png");
    });
}

export async function saveSDFToPNG(imageData, width, height, fileName) {
	// Step 4: Create a canvas and draw the image data
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d");
	const imageDataObj = ctx.createImageData(width, height);
	let index = 0;
	for (let i = 0; i < imageData.length; i++) {
		// BGRA to RGBA
		imageDataObj.data[index] = 255;
		imageDataObj.data[index + 1] = 255;
		imageDataObj.data[index + 2] = 255;
		imageDataObj.data[index + 3] = imageData[i] * 255;
		index += 4;
	}
	ctx.putImageData(imageDataObj, 0, 0);

	// Step 5: Convert canvas content to PNG and trigger download
	const dataURL = canvas.toDataURL("image/png");
	const a = document.createElement("a");
	a.href = dataURL;
	a.download = fileName;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
}

export async function saveMaskToPNG(textureData, width, height, fileName) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(width, height);
    
    // Convert R32Uint to RGBA
    let index = 0;
    for (let i = 0; i < textureData.length; i += 4) {
        const value = textureData[i]; // Only using first byte (assuming values are 0 or 1)
        imageData.data[index] = value * 255;
        imageData.data[index + 1] = value * 255;
        imageData.data[index + 2] = value * 255;
        imageData.data[index + 3] = 255; // Full alpha
        index += 4;
    }
    
    ctx.putImageData(imageData, 0, 0);
    const dataURL = canvas.toDataURL("image/png");
    const a = document.createElement("a");
    a.href = dataURL;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

async function createConversionPipeline(device) {
    const convertCode = await loadShader("convert.wgsl");
    const convertModule = device.createShaderModule({ code: convertCode });
    return device.createComputePipeline({
        compute: { module: convertModule, entryPoint: "main" },
        layout: "auto",
    });
}

export async function convertTexture(device, width, height, sourceTexture, targetTexture) {
    const conversionPipeline = await createConversionPipeline(device);
    const commandEncoder = device.createCommandEncoder();
    
    const bindGroup = device.createBindGroup({
        layout: conversionPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: sourceTexture.createView() },
            { binding: 1, resource: targetTexture.createView() },
        ],
    });

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(conversionPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(
        Math.ceil(width / 8),
        Math.ceil(height / 8)
    );
    pass.end();

    device.queue.submit([commandEncoder.finish()]);
}