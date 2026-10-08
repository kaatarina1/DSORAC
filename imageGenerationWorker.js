import { cleanup, initRenderer, renderImage } from "./js/CaptureRenderer.js";

const onProgress = (progress) => self.postMessage({
                type: 'PROGRESS',
                ...progress
            });

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        if (type === 'INIT') {
            await initRenderer(data);
            self.postMessage({ type: "INIT_COMPLETE" });
        }  else if (type === 'GENERATE_IMAGE') {
            const { imageIndex, rgba, metadata } = await renderImage(data, { onProgress });
            const blob = await rgbaToPngBlob(rgba, metadata.width, metadata.height);
            self.postMessage({ type: "IMAGE_COMPLETE", imageIndex, blob, metadata });
        } else if (type === 'SHUTDOWN') {
            cleanup();
            self.close();
        }
    } catch (error) {
        console.error(error.cause ?? error);
        self.postMessage({
            type: "ERROR",
            imageIndex: data?.imageIndex,
            error: error.message,
            stack: (error.cause ?? error).stack,
        });
    }
};

async function rgbaToPngBlob(data, width, height) {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d");

    const imgData = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
    
    ctx.putImageData(new ImageData(imgData, width, height), 0, 0);

    return await canvas.convertToBlob({ type: "image/png" });
}