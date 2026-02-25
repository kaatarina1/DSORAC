import { DepthMap } from "./js/DepthMap.js";
import { Composer } from "./js/Composer.js";
import { SignedDistanceFiled } from "./js/SignedDistanceFiled.js";
import { Solver } from "./js/Solvers.js";
import { convertTexture } from "./js/Utils.js";
import { LasLoader } from "./js/LasLoader.js";
import * as mat4 from "./js/mat4.js";
import { CameraPosition } from "./js/CameraPosition.js";


// Worker-safe ustvarjanje Blob objekta iz GPU teksture
async function getBlobWorkerSafe(data, width, height, colorOrder = "bgr") {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    
    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data;
    
    
    let maxVal = 0;
    let minVal = Infinity;
    let nonZeroCount = 0;
    
    for (let i = 0; i < data.length; i++) {
        const val = data[i];
        if (val !== 0) nonZeroCount++;
        maxVal = Math.max(maxVal, val);
        minVal = Math.min(minVal, val);
    }
    
    console.log(`📊 Image data: min=${minVal.toFixed(4)}, max=${maxVal.toFixed(4)}, nonZero=${nonZeroCount}/${data.length}`);
    
    if (maxVal === 0) {
        console.error('❌ WARNING: Output data is all zeros!');
    }
    
    // Preveri, ali so podatki že v 0-255 obsegu ali v 0-1 obsegu
    const needsScaling = maxVal <= 1.0;
    
    if (needsScaling) {
        console.log('🎨 Converting from 0-1 range to 0-255');
        for (let i = 0; i < width * height; i++) {
            const r = Math.max(0, Math.min(255, Math.floor(data[i * 4] * 255)));
            const g = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 1] * 255)));
            const b = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 2] * 255)));
            const a = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 3] * 255)));

            // Podpora za RGB ali BGR vrstni red barv v odvisnosti od vhodnih podatkov
            if (colorOrder === "rgb") {
                pixels[i * 4] = r;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = b;
            } else {
                pixels[i * 4] = b;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = r;
            }
            pixels[i * 4 + 3] = a;
        }
    } else {
        console.log('🎨 Data already in 0-255 range, copying directly');
        for (let i = 0; i < width * height; i++) {
            const r = Math.max(0, Math.min(255, Math.floor(data[i * 4])));
            const g = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 1])));
            const b = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 2])));
            const a = Math.max(0, Math.min(255, Math.floor(data[i * 4 + 3])));

            // Podpora za RGB ali BGR vrstni red barv v odvisnosti od vhodnih podatkov
            if (colorOrder === "rgb") {
                pixels[i * 4] = r;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = b;
            } else {
                pixels[i * 4] = b;
                pixels[i * 4 + 1] = g;
                pixels[i * 4 + 2] = r;
            }
            pixels[i * 4 + 3] = a;
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
    return await canvas.convertToBlob({ type: 'image/png' });
}

let device = null;
let renderingPipeline = null;
let pointclouds = [];
let canvas = null;
let format = null;
let deviceLost = false;
let deviceLostPromise = null;

// Bounding box oblaka točk (izračunano enkrat med inicializacijo)
let bbMin = [Infinity, Infinity, Infinity];
let bbMax = [-Infinity, -Infinity, -Infinity];

self.onmessage = async (e) => {
    const { type, data } = e.data;

    try {
        switch (type) {
            case 'INIT':
                await initializeWorker(data);
                break;
            case 'GENERATE_IMAGE':
                await generateImage(data);
                break;
            case 'SHUTDOWN':
                cleanup();
                break;
        }
    } catch (error) {
        self.postMessage({
            type: 'ERROR',
            error: error.message,
            stack: error.stack
        });
    }
};

async function initializeWorker(config) {
    try {
        console.log('🔧 Worker initializing GPU...');
        const adapter = await navigator.gpu.requestAdapter();
        const hasBGRA8unormStorage = adapter.features.has("bgra8unorm-storage");
        
        device = await adapter?.requestDevice({
            requiredFeatures: hasBGRA8unormStorage 
                ? ["bgra8unorm-storage", "float32-filterable", "float32-blendable", "timestamp-query"] 
                : ["float32-filterable", "float32-blendable", "timestamp-query"],
        });

        // Spremljamo izgubo GPU
        deviceLostPromise = device.lost.then((info) => {
            deviceLost = true;
            console.error('🔥 GPU Device Lost:', info.message, info.reason);
        });

        format = hasBGRA8unormStorage
            ? navigator.gpu.getPreferredCanvasFormat()
            : "rgba8unorm";

        canvas = new OffscreenCanvas(config.width, config.height);
        const context = canvas.getContext("webgpu");
        context.configure({ device, format, alphaMode: "premultiplied" });

        console.log('🔧 Worker creating rendering pipeline...');
        const renderingCode = await fetch("./shaders/rendering.wgsl").then(r => r.text());
        const renderingModule = device.createShaderModule({ code: renderingCode });

        renderingPipeline = device.createRenderPipeline({
            vertex: { module: renderingModule },
            fragment: {
                module: renderingModule,
                targets: [{ format }],
            },
            primitive: { topology: "point-list" },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth32float",
            },
            layout: "auto",
        });

        console.log("🔧 Worker loading point cloud...");
        const lasLoader = new LasLoader("./data/cropped_filtered_1.las");
        const lasData = await lasLoader.loadLasData();
        
        console.log(`📊 Worker loaded ${lasData.positions.length / 3} points`);
        
        // Izračunamo bounding box oblaka točk, 
        // da ga lahko uporabimo za nastavitev near/far 
        // ravnin kamere in za normalizacijo globin
        const positions = lasData.positions;
        bbMin = [Infinity, Infinity, Infinity];
        bbMax = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < positions.length; i += 3) {
            bbMin[0] = Math.min(bbMin[0], positions[i]);
            bbMin[1] = Math.min(bbMin[1], positions[i + 1]);
            bbMin[2] = Math.min(bbMin[2], positions[i + 2]);
            bbMax[0] = Math.max(bbMax[0], positions[i]);
            bbMax[1] = Math.max(bbMax[1], positions[i + 1]);
            bbMax[2] = Math.max(bbMax[2], positions[i + 2]);
        }
        console.log("📐 Point cloud bounding box:", bbMin, bbMax);
        
        await createPointClouds(positions, lasData.colors);
        
        console.log(`✅ Worker ready with ${pointclouds.length} point cloud batches`);

        self.postMessage({ type: 'INIT_COMPLETE' });
    } catch (error) {
        console.error('❌ Worker init error:', error);
        self.postMessage({
            type: 'ERROR',
            error: `Init failed: ${error.message}`,
            stack: error.stack
        });
    }
}

async function createPointClouds(positions, colors) {
    pointclouds = [];
    const pointByteSize = 16;
    const maxPointsPerBuffer = 1024 * 1024;
    const numberOfAllPoints = positions.length / 3;
    const numberOfBatches = Math.ceil(numberOfAllPoints / maxPointsPerBuffer);

    for (let i = 0; i < numberOfBatches; i++) {
        const startIndex = i * maxPointsPerBuffer;
        const numberOfPoints = Math.min(maxPointsPerBuffer, numberOfAllPoints - startIndex);

        const pointData = new ArrayBuffer(numberOfPoints * pointByteSize);
        const pointDataView = new DataView(pointData);

        for (let j = 0; j < numberOfPoints; j++) {
            const posIndex = (startIndex + j) * 3;
            const pointOffset = j * pointByteSize;

            pointDataView.setFloat32(pointOffset, positions[posIndex], true);
            pointDataView.setFloat32(pointOffset + 4, positions[posIndex + 1], true);
            pointDataView.setFloat32(pointOffset + 8, positions[posIndex + 2], true);
            pointDataView.setUint32(pointOffset + 12, colors[startIndex + j], true);
        }

        const pointBuffer = device.createBuffer({
            size: numberOfPoints * pointByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(pointBuffer, 0, pointData);

        const renderingBindGroup = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: pointBuffer } }],
        });

        pointclouds.push({
            pointBuffer,
            renderingBindGroup,
            numberOfPoints,
        });
    }
}

function createDepthRangeBindGroup(minDepth, maxDepth) {
    const buf = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([minDepth, maxDepth]));
    return {
        buffer: buf,
        bindGroup: device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    };
}

function createTargetPositionBindGroup(pos) {
    const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([pos[0], pos[1], pos[2], 0.0]));
    return {
        buffer: buf,
        bindGroup: device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    };
}

// Combined MVP + view matrix in group(1): binding(0)=MVP, binding(1)=viewMatrix
function createMatricesBindGroup(projectionViewMatrix, viewMatrix) {
    const mvpBuf = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(mvpBuf, 0, new Float32Array(projectionViewMatrix));

    const viewBuf = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(viewBuf, 0, new Float32Array(viewMatrix));

    return {
        mvpBuffer: mvpBuf,
        viewBuffer: viewBuf,
        bindGroup: device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: mvpBuf } },
                { binding: 1, resource: { buffer: viewBuf } },
            ],
        }),
    };
}

async function generateImage(params) {
    const imageIndex = params.imageIndex;
    const camPositionHelper = new CameraPosition(canvas, device, renderingPipeline);
    try {
        const {
            cameraPosition,
            target,
            targetPosition,
            useReconstruction = true,
            colorOrder = "bgr",
        } = params;

        // Preveri stanje GPU naprave
        if (deviceLost || !device) {
            throw new Error('GPU device is lost or invalid');
        }

        // Zagotovimo da je GPU pripravljen
        await device.queue.onSubmittedWorkDone();

        // Izračunamo view in projection matriki
        let { viewMatrix, projectionViewMatrix, near, far } =
            camPositionHelper.computeCameraMatrix(cameraPosition, target, canvas, bbMin, bbMax);

        const depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: "depth32float",
        });

        const { bindGroup: matricesBindGroup, mvpBuffer, viewBuffer } =
            createMatricesBindGroup(projectionViewMatrix, viewMatrix);

        let outputData;

        if (useReconstruction) {
            await renderFullScene(matricesBindGroup, depthTexture, targetPosition);

            const depthMap = new DepthMap(canvas, device, depthTexture);
            let depthBins;
            
            try {
                depthBins = await depthMap.groupDepthIntoBins({ near, far });
                if (!depthBins || depthBins.length === 0) {
                    depthBins = [[0, 0.3], [0.3, 0.6], [0.6, 1.0]];
                }
                depthBins.reverse();
            } catch (error) {
                depthBins = [[0, 0.3], [0.3, 0.6], [0.6, 1.0]];
            }

            self.postMessage({
                type: 'PROGRESS',
                imageIndex,
                message: `Calculated ${depthBins.length} depth bins (near=${near.toFixed(4)}, far=${far.toFixed(4)})`
            });

            // Za vsako sliko naredimo novo instanco composer razreda
            const composer = new Composer(self, canvas, device, format);
            composer.initializeCompositeTexture();
            
            // Zagotovimo, da Composer začne s čistimi podatki
            composer.reconstructions = [];
            composer.sdfs = [];
            composer.depthPoints = [];
            composer.depths = [];
            
            console.log(`🎨 Starting composition with ${depthBins.length} depth bins`);

            for (let i = 0; i < depthBins.length; i++) {
                const [minDepth, maxDepth] = depthBins[i];
                await renderPointsInDepthRange(
                    minDepth, maxDepth, near, far,
                    matricesBindGroup, depthTexture, composer,
                    targetPosition
                );
                
                console.log(`✓ Layer ${i + 1}/${depthBins.length} added. Composer now has ${composer.reconstructions.length} layers`);
                
                self.postMessage({
                    type: 'PROGRESS',
                    imageIndex,
                    layer: i,
                    totalLayers: depthBins.length
                });
            }
            
            console.log(`🎬 Compositing ${composer.reconstructions.length} layers...`);

            // Zagotovimo, da so vsi GPU ukazi zaključeni pred kompozicijo
            await device.queue.onSubmittedWorkDone();
            
            outputData = await composer.compositeDepths(composer);

            if (composer.compositeResult) {
                composer.compositeResult.destroy();
            }
        } else {
            self.postMessage({
                type: 'PROGRESS',
                imageIndex,
                message: 'Capturing raw point cloud (reconstruction disabled)'
            });
            outputData = await capturePointCloudImage(matricesBindGroup, depthTexture, targetPosition);
        }

        const blob = await getBlobWorkerSafe(outputData, canvas.width, canvas.height, colorOrder);
        
        //Čakamo, da se vse GPU operacije zaključijo
        await device.queue.onSubmittedWorkDone();
        
        const { quat, t } = mat4.cameraToColmapPose(cameraPosition, viewMatrix);
        const { fx, fy, cx, cy, width, height } = mat4.getIntrinsics(
            canvas.width,
            canvas.height,
            45 * Math.PI / 180
        );

        // Zadnja sinhronizacija GPU pred čiščenjem
        await device.queue.onSubmittedWorkDone();
        
        depthTexture.destroy();
        mvpBuffer.destroy();
        viewBuffer.destroy();
        
        self.postMessage({
            type: 'IMAGE_COMPLETE',
            imageIndex,
            blob,
            metadata: {
                quat, t, fx, fy, cx, cy, width, height,
                filename: `image_${imageIndex + 1}_${target.join("_")}.png`
            }
        });

        console.log(`✅ Image ${imageIndex} completed successfully`);

    } catch (error) {
        console.error(`❌ Error generating image ${imageIndex}:`, error);
        self.postMessage({
            type: 'ERROR',
            imageIndex: imageIndex,
            error: error.message,
            stack: error.stack
        });
    }
}

async function renderFullScene(matricesBindGroup, depthTexture, targetPosition) {
    const { bindGroup: depthRangeBG, buffer: drBuf } =
        createDepthRangeBindGroup(0, 1e6);

    const { bindGroup: targetBG, buffer: tBuf } =
        createTargetPositionBindGroup(targetPosition);

    const tempTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        format: format,
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: tempTexture.createView(),
            loadOp: "clear",
            clearValue: [0, 0, 0, 0],
            storeOp: "store",
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear",
            depthClearValue: 1,
            depthStoreOp: "store",
        },
    });

    renderPass.setPipeline(renderingPipeline);
    renderPass.setBindGroup(1, matricesBindGroup);
    renderPass.setBindGroup(2, depthRangeBG);
    renderPass.setBindGroup(3, targetBG);

    for (const pointcloud of pointclouds) {
        renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
        renderPass.draw(pointcloud.numberOfPoints);
    }

    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    tempTexture.destroy();
    drBuf.destroy();
    tBuf.destroy();
}

async function renderPointsInDepthRange(
    minDepth, maxDepth, near, far,
    matricesBindGroup, depthTexture, composer,
    targetPosition
) {
    const { bindGroup: depthRangeBG, buffer: drBuf } =
        createDepthRangeBindGroup(minDepth, maxDepth);

    const { bindGroup: targetBG, buffer: tBuf } =
        createTargetPositionBindGroup(targetPosition);

    const captureTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_SRC |
               GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.RENDER_ATTACHMENT,
        format: format,
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: captureTexture.createView(),
            loadOp: "clear",
            clearValue: [0, 0, 0, 0],
            storeOp: "store",
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear",
            depthClearValue: 1,
            depthStoreOp: "store",
        },
    });

    renderPass.setPipeline(renderingPipeline);
    renderPass.setBindGroup(1, matricesBindGroup);
    renderPass.setBindGroup(2, depthRangeBG);
    renderPass.setBindGroup(3, targetBG);

    for (const pointcloud of pointclouds) {
        renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
        renderPass.draw(pointcloud.numberOfPoints);
    }

    renderPass.end();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();  

    let reconstructionRead = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.COPY_SRC,
    });

    await convertTexture(device, canvas.width, canvas.height, captureTexture, reconstructionRead);

    let reconstructionWrite = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.COPY_SRC,
    });

    const sdf = new SignedDistanceFiled(device, captureTexture, canvas.width, canvas.height);
    const sdfTexture = await sdf.generateSDF();

    let solver = new Solver(canvas, device);
    await solver.sorRedBlack(captureTexture, reconstructionRead, reconstructionWrite);

    let pointsTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: "rgba32float",
        usage: GPUTextureUsage.STORAGE_BINDING |
               GPUTextureUsage.TEXTURE_BINDING |
               GPUTextureUsage.COPY_DST |
               GPUTextureUsage.COPY_SRC,
    });

    await convertTexture(device, canvas.width, canvas.height, captureTexture, pointsTexture);

    await composer.addLayers(sdfTexture, reconstructionRead, pointsTexture, (minDepth + maxDepth) / 2);
    
    await device.queue.onSubmittedWorkDone();

    captureTexture.destroy();
    reconstructionRead.destroy();
    reconstructionWrite.destroy();
    sdfTexture.destroy();
    pointsTexture.destroy();
    drBuf.destroy();
    tBuf.destroy();
}

async function capturePointCloudImage(matricesBindGroup, depthTexture, targetPosition) {
    const { bindGroup: depthRangeBG, buffer: drBuf } =
        createDepthRangeBindGroup(0, 1e6);

    const { bindGroup: targetBG, buffer: tBuf } =
        createTargetPositionBindGroup(targetPosition);

    const captureTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.COPY_SRC |
               GPUTextureUsage.RENDER_ATTACHMENT,
        format: format,
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: captureTexture.createView(),
            loadOp: "clear",
            clearValue: [0, 0, 0, 0],
            storeOp: "store",
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear",
            depthClearValue: 1,
            depthStoreOp: "store",
        },
    });

    renderPass.setPipeline(renderingPipeline);
    renderPass.setBindGroup(1, matricesBindGroup);
    renderPass.setBindGroup(2, depthRangeBG);
    renderPass.setBindGroup(3, targetBG);

    for (const pointcloud of pointclouds) {
        renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
        renderPass.draw(pointcloud.numberOfPoints);
    }

    renderPass.end();

    const bytesPerPixel = 4;
    const alignedBytesPerRow = Math.ceil((canvas.width * bytesPerPixel) / 256) * 256;
    const outputBuffer = device.createBuffer({
        size: alignedBytesPerRow * canvas.height,
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
            bytesPerRow: alignedBytesPerRow,
            rowsPerImage: canvas.height,
        },
        [canvas.width, canvas.height, 1]
    );

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await outputBuffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(outputBuffer.getMappedRange());
    const tightData = new Uint8Array(canvas.width * canvas.height * bytesPerPixel);

    for (let y = 0; y < canvas.height; y++) {
        const srcOffset = y * alignedBytesPerRow;
        const dstOffset = y * canvas.width * bytesPerPixel;
        tightData.set(
            mapped.subarray(srcOffset, srcOffset + canvas.width * bytesPerPixel),
            dstOffset
        );
    }

    outputBuffer.unmap();
    outputBuffer.destroy();
    captureTexture.destroy();
    drBuf.destroy();
    tBuf.destroy();

    return tightData;
}

function cleanup() {
    if (device) {
        for (const pc of pointclouds) {
            pc.pointBuffer.destroy();
        }
        pointclouds = [];
    }
    self.close();
}