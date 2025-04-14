import { Buffer } from "buffer";

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

export async function saveTextureToPNG(imageData, width, height) {
	// Step 4: Create a canvas and draw the image data
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d");
	const imageDataObj = ctx.createImageData(width, height);

	// Copy the pixel data to the ImageData object
	for (let i = 0; i < imageData.length; i += 4) {
		// BGRA to RGBA
		imageDataObj.data[i] = imageData[i + 2];
		imageDataObj.data[i + 1] = imageData[i + 1];
		imageDataObj.data[i + 2] = imageData[i];
		imageDataObj.data[i + 3] = imageData[i + 3];
	}
	ctx.putImageData(imageDataObj, 0, 0);

	// Step 5: Convert canvas content to PNG and trigger download
	const dataURL = canvas.toDataURL("image/png");
	const a = document.createElement("a");
	a.href = dataURL;
	a.download = "capture.png";
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
}
