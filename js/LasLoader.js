import { createLazPerf } from "laz-perf";
import { parseHeader } from "./Utils";

export class LasLoader {
	constructor(filename) {
        this.lasName = filename;
	}

    async loadLasData() {
        const LazPerf = await createLazPerf({
            locateFile: (file) => `./node_modules/laz-perf/lib/laz-perf.wasm`,
        });
        const response = await fetch(this.lasName);
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

        return {positions, colors}
    }
}