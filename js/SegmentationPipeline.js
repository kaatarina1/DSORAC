import * as mat4 from "./mat4.js";
import JSZip from "jszip";

// PNG DECODER (16-bit grayscale, za segmentacijske maske)
function paethPredictor(a, b, c) {
    const p = a + b - c;
    const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    return (pa <= pb && pa <= pc) ? a : (pb <= pc) ? b : c;
}

// Dekodira sivinsko sliko (v kateri so zapisani razredi) in vrne Uint16Array vrednosti pikslov
async function decodeMaskPng(arrayBuffer) {
    const bytes = new Uint8Array(arrayBuffer);
    const dv    = new DataView(arrayBuffer);

    let pos = 8; // skip 8-byte PNG signature
    let W = 0, H = 0, bitDepth = 0, colorType = 0;
    const idatParts = [];

    while (pos + 8 <= bytes.length) {
        const length = dv.getUint32(pos); pos += 4;
        const type   = String.fromCharCode(bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]); pos += 4;

        if (type === 'IHDR') {
            W = dv.getUint32(pos); H = dv.getUint32(pos + 4);
            bitDepth = bytes[pos + 8]; colorType = bytes[pos + 9];
        } else if (type === 'IDAT') {
            idatParts.push(bytes.slice(pos, pos + length));
        } else if (type === 'IEND') {
            break;
        }
        pos += length + 4; // skip data + CRC
    }

    // 8-bit sivinska slika: canvas API (vrednosti 0-255 se ohranijo)
    if (bitDepth === 8 && colorType === 0) {
        const blob = new Blob([arrayBuffer], { type: 'image/png' });
        const bmp  = await createImageBitmap(blob);
        const oc   = new OffscreenCanvas(W, H);
        oc.getContext('2d').drawImage(bmp, 0, 0);
        const imgd = oc.getContext('2d').getImageData(0, 0, W, H).data;
        const out  = new Uint16Array(W * H);
        for (let i = 0; i < out.length; i++) out[i] = imgd[i * 4];
        return out;
    }

    // 16-bit: najprej je potrebno dekompresirati IDAT 
    const totalLen = idatParts.reduce((s, p) => s + p.length, 0);
    const idat = new Uint8Array(totalLen);
    let off = 0;
    for (const p of idatParts) { idat.set(p, off); off += p.length; }

    const ds = new DecompressionStream('deflate');
    const writer = ds.writable.getWriter();
    const reader = ds.readable.getReader();
    writer.write(idat);
    writer.close();

    const rawChunks = [];
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        rawChunks.push(value);
    }
    const raw = new Uint8Array(rawChunks.reduce((s, c) => s + c.length, 0));
    off = 0;
    for (const c of rawChunks) { raw.set(c, off); off += c.length; }

    const bps  = bitDepth === 16 ? 2 : 1; // bytes per sample
    const rowB = W * bps;
    const pixels  = new Uint16Array(W * H);
    const prevRow = new Uint8Array(rowB);

    for (let y = 0; y < H; y++) {
        const base   = y * (rowB + 1);
        const filter = raw[base];
        const src    = raw.subarray(base + 1, base + 1 + rowB);
        const cur    = new Uint8Array(rowB);

        for (let i = 0; i < rowB; i++) {
            const v = src[i];
            const a = i >= bps ? cur[i - bps] : 0;
            const b = prevRow[i];
            const c = i >= bps ? prevRow[i - bps] : 0;
            switch (filter) {
                case 0: cur[i] = v; break;
                case 1: cur[i] = (v + a) & 0xFF; break;
                case 2: cur[i] = (v + b) & 0xFF; break;
                case 3: cur[i] = (v + ((a + b) >> 1)) & 0xFF; break;
                case 4: cur[i] = (v + paethPredictor(a, b, c)) & 0xFF; break;
                default: cur[i] = v;
            }
        }
        prevRow.set(cur);

        for (let x = 0; x < W; x++) {
            pixels[y * W + x] = bitDepth === 16
                ? (cur[x * 2] << 8) | cur[x * 2 + 1]
                : cur[x];
        }
    }

    return pixels;
}

// SEGMENTATION PIPELINE
export class SegmentationPipeline {
    device;
    numberOfAllPoints;
    pointclouds;
    classPalette;
    controls;

    constructor({ device, numberOfAllPoints, pointclouds, classPalette, controls }) {
        this.device = device;
        this.numberOfAllPoints = numberOfAllPoints;
        this.pointclouds = pointclouds;
        this.classPalette = classPalette;
        this.controls = controls;
    }

    // Iz COLMAP quaternionov in translacij dobimo view matriko
    // COLMAP: y-down, z-forward. WebGPU: y-up, z-backward.
    colmapQuatToViewMatrix(qw, qx, qy, qz, tx, ty, tz) {
        const R00 = 1 - 2*(qy*qy + qz*qz), R01 = 2*(qx*qy - qz*qw), R02 = 2*(qx*qz + qy*qw);
        const R10 = 2*(qx*qy + qz*qw),     R11 = 1 - 2*(qx*qx + qz*qz), R12 = 2*(qy*qz - qx*qw);
        const R20 = 2*(qx*qz - qy*qw),     R21 = 2*(qy*qz + qx*qw),     R22 = 1 - 2*(qx*qx + qy*qy);
        // FlipYZ (negira vrstici 1 in 2)
        // Pretvori COLMAP prostor kamere v WebGPU prostor kamere
        return new Float32Array([
             R00, -R10, -R20, 0,
             R01, -R11, -R21, 0,
             R02, -R12, -R22, 0,
             tx,  -ty,  -tz,  1,
        ]);
    }

    // Zgradi perspektivno projekcijsko matriko iz  COLMAP PINHOLE intrinsics
    // Izpeljani tako, da se (u, v) koordinate pikslov ujemamjo s COLMAP projekcijo
    colmapIntrinsicsToProjection(fx, fy, cx, cy, W, H, near, far) {
        const P = new Float32Array(16);
        P[0]  =  2*fx/W;
        P[5]  =  2*fy/H;
        P[8]  =  1 - 2*cx/W;
        P[9]  =  2*cy/H - 1;
        P[10] =  far / (near - far);
        P[11] = -1;
        P[14] =  far * near / (near - far);
        return P;
    }

    async processZip(zipFile) {
        this.controls.setSegmentationStatus('Reading ZIP...');

        let zipData;
        try {
            zipData = await JSZip.loadAsync(zipFile);
        } catch {
            this.controls.setSegmentationStatus('Error during reding ZIP.');
            return;
        }

        // Find files by suffix (handle nested paths)
        const find = (suffix) => Object.keys(zipData.files).find(f => f.endsWith(suffix));

        const camerasKey      = find('cameras.txt');
        const imagesKey       = find('images.txt');
        const classMappingKey = find('class_mapping.json');

        if (!camerasKey || !imagesKey) {
            this.controls.setSegmentationStatus('ZIP must contain cameras.txt and images.txt.');
            return;
        }

        const camerasText  = await zipData.files[camerasKey].async('string');
        const imagesText   = await zipData.files[imagesKey].async('string');
        const classMapping = classMappingKey
            ? JSON.parse(await zipData.files[classMappingKey].async('string'))
            : null;

        // Parse cameras.txt: ID PINHOLE W H fx fy cx cy
        const cameras = {};
        for (const line of camerasText.split('\n')) {
            const l = line.trim();
            if (l.startsWith('#') || !l) continue;
            const p = l.split(/\s+/);
            if (p[1] === 'PINHOLE') {
                cameras[p[0]] = {
                    width: Number(p[2]), height: Number(p[3]),
                    fx: Number(p[4]), fy: Number(p[5]),
                    cx: Number(p[6]), cy: Number(p[7]),
                };
            }
        }

        // Parse images.txt: ID QW QX QY QZ TX TY TZ CAM_ID NAME [\n POINTS2D]
        const images = {};
        const lines = imagesText.split('\n');
        for (let i = 0; i < lines.length; i++) {
            const l = lines[i].trim();
            if (l.startsWith('#') || !l) continue;
            const p = l.split(/\s+/);
            if (p.length >= 10) {
                images[p[9]] = {
                    qw: Number(p[1]), qx: Number(p[2]), qy: Number(p[3]), qz: Number(p[4]),
                    tx: Number(p[5]), ty: Number(p[6]), tz: Number(p[7]),
                    cameraId: p[8],
                };
                i++; // preskočimo POINTS2D vrstico
            }
        }

        // Pridobimo slike s segmentacijskimi maskami
        const maskFiles = Object.keys(zipData.files)
            .filter(f => {
                const l = f.toLowerCase();
                return (l.endsWith('.png') || l.endsWith('.jpg')) && !zipData.files[f].dir;
            })
            .sort();

        // Ohranimo le makse vsebovane v image.txt ter z znano kamero
        const matchedMasks = maskFiles.filter(f => {
            const name = f.split('/').pop();
            return images[name] && cameras[images[name].cameraId];
        });

        console.log(`Segmentation ZIP: ${Object.keys(cameras).length} camera(s), ${Object.keys(images).length} images, ${maskFiles.length} masks (${matchedMasks.length} matched)`);
        if (classMapping) console.log(`Class mapping: ${Object.keys(classMapping).length} classes`);

        if (matchedMasks.length === 0) {
            this.controls.setSegmentationStatus(`Non of the masks matches (${maskFiles.length} mask, ${Object.keys(images).length} inputs).`);
            return;
        }

        this.controls.setSegmentationStatus(`${matchedMasks.length}/${maskFiles.length} masks match — starting proccessing...`);

        await this.runSegmentation(zipData, cameras, images, matchedMasks, classMapping);
    }

    async runSegmentation(zipData, cameras, images, matchedMasks, classMapping) {
        const device = this.device;
        const numberOfAllPoints = this.numberOfAllPoints;
        const pointclouds = this.pointclouds;
        const classPalette = this.classPalette;
        const controls = this.controls;

        const MASK_W = 1024, MASK_H = 512;
        const NEAR = 0.01, FAR = 500.0;

        // Določi N_CLASSES iz preslikave razredov (max class ID + 1, minimum 128)
        let N_CLASSES = 128;
        if (classMapping) {
            const flatMap = classMapping.label_to_id ?? classMapping;
            const ids = Object.values(flatMap).filter(v => Number.isInteger(v));
            if (ids.length > 0) N_CLASSES = Math.max(128, Math.max(...ids) + 1);
        }
        console.log(`N_CLASSES=${N_CLASSES}, votes buffer=${(numberOfAllPoints * N_CLASSES * 4 / 1024 / 1024).toFixed(0)} MB`);

        const [indexDiskCode, voteCode] = await Promise.all([
            fetch('./shaders/indexDisk.wgsl').then(r => r.text()),
            fetch('./shaders/vote.wgsl').then(r => r.text()),
        ]);

        const indexDiskModule = device.createShaderModule({ code: indexDiskCode });
        const indexDiskPipeline = device.createRenderPipeline({
            vertex:   { module: indexDiskModule },
            fragment: { module: indexDiskModule, targets: [{ format: 'r32uint' }] },
            primitive: { topology: 'triangle-list', cullMode: 'none' },
            depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth32float' },
            layout: 'auto',
        });

        const votePipeline = device.createComputePipeline({
            compute: { module: device.createShaderModule({ code: voteCode }), entryPoint: 'vote' },
            layout: 'auto',
        });

        // Index texture: r32uint, potrebuje TEXTURE_BINDING za vote compute shader
        const indexTexture = device.createTexture({
            size: [MASK_W, MASK_H],
            format: 'r32uint',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        const indexDepthTexture = device.createTexture({
            size: [MASK_W, MASK_H],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Votes buffer: numberOfAllPoints × N_CLASSES u32s, zero-initialised
        const votesBuffer = device.createBuffer({
            size: numberOfAllPoints * N_CLASSES * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        {
            const enc = device.createCommandEncoder();
            enc.clearBuffer(votesBuffer);
            device.queue.submit([enc.finish()]);
        }

        // Mask pixel buffer (en u32 na piksl, prepisano za vsako masko)
        const maskBuffer = device.createBuffer({
            size: MASK_W * MASK_H * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Vote shader parametri: nClasses, width, height, unused
        const voteParamsBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(voteParamsBuffer, 0, new Uint32Array([N_CLASSES, MASK_W, MASK_H, 0]));

        // Index-disk scene uniform: mat4 MVP (64 bytes) + vec4 {imgW, imgH, radius, pad} (16 bytes) = 80 bytes
        const DISK_RADIUS_PX = 4.0; // screen-space radij diska
        const indexSceneBuffer = device.createBuffer({
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const indexCellBGs = pointclouds.map(pc => device.createBindGroup({
            layout: indexDiskPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: pc.pointBuffer } }],
        }));
        const indexSceneBG = device.createBindGroup({
            layout: indexDiskPipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: indexSceneBuffer } }],
        });
        const voteBG = device.createBindGroup({
            layout: votePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: votesBuffer } },
                { binding: 1, resource: indexTexture.createView() },
                { binding: 2, resource: { buffer: maskBuffer } },
                { binding: 3, resource: { buffer: voteParamsBuffer } },
            ],
        });

        let processed = 0;

        for (const maskPath of matchedMasks) {
            const name = maskPath.split('/').pop();
            const img  = images[name];
            const cam  = cameras[img.cameraId];

            // Zgradi MVP za COLMAP quaternione in translacije
            const { qw, qx, qy, qz, tx, ty, tz } = img;
            const V   = this.colmapQuatToViewMatrix(qw, qx, qy, qz, tx, ty, tz);
            const P   = this.colmapIntrinsicsToProjection(cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height, NEAR, FAR);
            const MVP = mat4.multiply(P, V);

            // mat4 MVP (16 floats) + {imgW, imgH, radius, pad} (4 floats) = 20 floats = 80 bytes
            const sceneData = new Float32Array(20);
            sceneData.set(MVP, 0);
            sceneData[16] = MASK_W;
            sceneData[17] = MASK_H;
            sceneData[18] = DISK_RADIUS_PX;
            sceneData[19] = 0;
            device.queue.writeBuffer(indexSceneBuffer, 0, sceneData);

            // Index render pass: zapiše indekse vidnih točk v indexTexture 
            {
                const enc   = device.createCommandEncoder();
                const rpass = enc.beginRenderPass({
                    colorAttachments: [{
                        view: indexTexture.createView(),
                        loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store',
                    }],
                    depthStencilAttachment: {
                        view: indexDepthTexture.createView(),
                        depthLoadOp: 'clear', depthClearValue: 1.0, depthStoreOp: 'store',
                    },
                });
                rpass.setPipeline(indexDiskPipeline);
                rpass.setBindGroup(1, indexSceneBG);
                for (let i = 0; i < pointclouds.length; i++) {
                    rpass.setBindGroup(0, indexCellBGs[i]);
                    rpass.draw(pointclouds[i].numberOfPoints * 6);
                }
                rpass.end();
                device.queue.submit([enc.finish()]);
                await device.queue.onSubmittedWorkDone();
            }

            // Dekodira maskp in naloži razrede v maskBuffer
            controls.setSegmentationStatus(`Decodinf ${processed + 1}/${matchedMasks.length}: ${name}`);
            const pngBytes = await zipData.files[maskPath].async('arraybuffer');
            const classIds = await decodeMaskPng(pngBytes);

            const maskU32 = new Uint32Array(MASK_W * MASK_H);
            for (let i = 0; i < classIds.length; i++) maskU32[i] = classIds[i];
            device.queue.writeBuffer(maskBuffer, 0, maskU32);

            // Vote compute pass — glasovanje
            {
                const enc   = device.createCommandEncoder();
                const cpass = enc.beginComputePass();
                cpass.setPipeline(votePipeline);
                cpass.setBindGroup(0, voteBG);
                cpass.dispatchWorkgroups(Math.ceil(MASK_W / 16), Math.ceil(MASK_H / 16));
                cpass.end();
                device.queue.submit([enc.finish()]);
                await device.queue.onSubmittedWorkDone();
            }

            processed++;
            controls.setSegmentationStatus(`Voting: ${processed}/${matchedMasks.length}`);
        }

        controls.setSegmentationStatus('Argmax...');

        const [argmaxCode, applyCode] = await Promise.all([
            fetch('./shaders/argmax.wgsl').then(r => r.text()),
            fetch('./shaders/applySegmentation.wgsl').then(r => r.text()),
        ]);

        const argmaxPipeline = device.createComputePipeline({
            compute: { module: device.createShaderModule({ code: argmaxCode }), entryPoint: 'argmax' },
            layout: 'auto',
        });
        const applyPipeline = device.createComputePipeline({
            compute: { module: device.createShaderModule({ code: applyCode }), entryPoint: 'apply' },
            layout: 'auto',
        });

        // Class color palette (256 zapisov, ki so določeni ob zagonu)
        const paletteBuffer = device.createBuffer({
            size: 256 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paletteBuffer, 0, classPalette);

        // Winner buffer: en classColor u32 na globalno točko
        const winnerBuffer = device.createBuffer({
            size: numberOfAllPoints * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Argmax params: nClasses, nPoints
        const argmaxParamsBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(argmaxParamsBuffer, 0, new Uint32Array([N_CLASSES, numberOfAllPoints, 0, 0]));

        const argmaxBG = device.createBindGroup({
            layout: argmaxPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: votesBuffer } },
                { binding: 1, resource: { buffer: paletteBuffer } },
                { binding: 2, resource: { buffer: winnerBuffer } },
                { binding: 3, resource: { buffer: argmaxParamsBuffer } },
            ],
        });

        // argmax: iz glasov določimo zmagovalni razred
        {
            const enc   = device.createCommandEncoder();
            const cpass = enc.beginComputePass();
            cpass.setPipeline(argmaxPipeline);
            cpass.setBindGroup(0, argmaxBG);
            cpass.dispatchWorkgroups(Math.ceil(numberOfAllPoints / 256));
            cpass.end();
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        const applyBGs = pointclouds.map(pc => {
            const nBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(nBuf, 0, new Uint32Array([pc.numberOfPoints, 0, 0, 0]));
            return device.createBindGroup({
                layout: applyPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pc.pointBuffer } },
                    { binding: 1, resource: { buffer: winnerBuffer } },
                    { binding: 2, resource: { buffer: nBuf } },
                ],
            });
        });

        // Na podlagi zmagovalnih razredov pripišemo id zmagovalnega razreda točki
        // objektu Point določimo atribut classColor
        {
            const enc   = device.createCommandEncoder();
            const cpass = enc.beginComputePass();
            cpass.setPipeline(applyPipeline);
            for (let i = 0; i < pointclouds.length; i++) {
                cpass.setBindGroup(0, applyBGs[i]);
                cpass.dispatchWorkgroups(Math.ceil(pointclouds[i].numberOfPoints / 256));
            }
            cpass.end();
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // Avtomatsko po določanju razredov zamenjamo pogled na takega
        // kjer so točke obarvane glede na razred kateremu pripadajo
        controls.enableClassification();
        controls.setSegmentationStatus(`Segmentation finished (${processed} views)!`, true);
    }
}
