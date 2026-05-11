import { DepthMap } from "./js/DepthMap.js";
import { Composer } from "./js/Composer.js";
import { SignedDistanceFiled } from "./js/SignedDistanceFiled.js";
import { Solver } from "./js/Solvers.js";
import { convertTexture } from "./js/Utils.js";
import { LasLoader } from "./js/LasLoader.js";
import * as mat4 from "./js/mat4.js";
import { CameraPosition } from "./js/CameraPosition.js";
import { AdaptiveGrid } from "./js/AdaptiveGrid.js";

let device = null;
let renderingPipeline = null;
let preparationPipeline = null;
let localSortPipeline = null;
let globalSortPipeline = null;
let quadPipelines = {};
let pointclouds = [];
let canvas = null;
// Dummy buffers za rendering.wgsl group-0 bindingi 1 & 2 (classColors / useClassColors).
// Worker zajema le RGB slike scene, se pravi nikoli ne upodablja scene z barvami razredov
let dummyClassColorsBuffer = null;
let useClassColorsBuffer   = null;
let format = null;
let deviceLost = false;
let deviceLostPromise = null;

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

// Bounding box oblaka točk (izračunano enkrat med inicializacijo)
let bbMin = [Infinity, Infinity, Infinity];
let bbMax = [-Infinity, -Infinity, -Infinity];
let numberOfAllPoints = 0;

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

        format = "rgba8unorm";

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
                targets: [{ format: format }],
            },
            primitive: { topology: "point-list" },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth32float",
            },
            layout: "auto",
        });

        // ostali pipeline-i (DISKS, BILLBOARDS, GAUSSIANS) 
        // delajo z istim formatom in bind group layouti
        const shaders = {
            DISKS: "./shaders/rendering_disks.wgsl",
            BILLBOARDS: "./shaders/rendering_billboards.wgsl",
            GAUSSIANS: "./shaders/rendering_gaussians.wgsl",
        };
        for (const [mode, shaderPath] of Object.entries(shaders)) {
            const code = await fetch(shaderPath).then(r => r.text());
            const module = device.createShaderModule({ code });
            const enableBlend = mode === "GAUSSIANS";
            const blendState = {
                color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            };
            quadPipelines[mode] = device.createRenderPipeline({
                vertex: { module },
                fragment: { module, targets: [{ format: format, ...(enableBlend ? { blend: blendState } : {}) }] },
                primitive: { topology: "triangle-list", cullMode: "none" },
                depthStencil: {
                    depthWriteEnabled: !enableBlend,
                    depthCompare: "less",
                    format: "depth32float"
                },
                layout: "auto",
            });
        }

        console.log("🔧 Worker creating sorting pipelines...");
        try {
            // Ustvarimo compute pipeline-e za pripravo podatkov in sortiranje, če so shaderji na voljo
            const preparationCode = await fetch("./shaders/preparation.wgsl").then(r => r.text());
            const localSortCode = await fetch("./shaders/localSort.wgsl").then(r => r.text());
            const globalSortCode = await fetch("./shaders/globalSort.wgsl").then(r => r.text());

            preparationPipeline = device.createComputePipeline({
                compute: { module: device.createShaderModule({ code: preparationCode }), entryPoint: "main" },
                layout: "auto",
            });
            localSortPipeline = device.createComputePipeline({
                compute: { module: device.createShaderModule({ code: localSortCode }), entryPoint: "compute" },
                layout: "auto",
            });
            globalSortPipeline = device.createComputePipeline({
                compute: { module: device.createShaderModule({ code: globalSortCode }), entryPoint: "mergePass" },
                layout: "auto",
            });
            console.log("✅ Sorting pipelines created successfully");
        } catch (error) {
            console.warn("⚠️ Warning: Sorting pipelines not available:", error.message);
            console.log("   Sorting will be skipped if requested");
        }

        // Dummy bufferji za rendering.wgsl group-0 bindingi 1 & 2.
        dummyClassColorsBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        useClassColorsBuffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(useClassColorsBuffer, 0, new Uint32Array([0]));

        console.log("🔧 Worker loading point cloud...");
        const lasLoader = new LasLoader("./data/pc/room_normals.las");
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

        await createPointClouds(positions, lasData.colors, lasData.normals);
        
        if (preparationPipeline && localSortPipeline && globalSortPipeline) {
            prewarmMergeBuffers(numberOfAllPoints);
            for (const pc of pointclouds) {
                buildSortBindGroups(pc);
            }
            console.log("✅ Sorting enabled");
        } else {
            console.log("ℹ️ Sorting pipelines not available");
        }

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

async function createPointClouds(positions, colors, normals) {
    pointclouds = [];
    const pointByteSize = 32;
    const maxPointsPerBuffer = 1024 * 1024;
    numberOfAllPoints = positions.length / 3;

    // Uporabimo adaptive grid za razdelitev točk v celice, 
    // kar omogoča renderiranje z gaussovkami in optimizacijo GPU pomnilnika.
    const grid = new AdaptiveGrid(positions, maxPointsPerBuffer);
    console.log(`📐 Worker adaptive grid: ${grid.cells.length} cells`);

    for (const cell of grid.cells) {
        const count = cell.indices.length;

        const pointData = new ArrayBuffer(maxPointsPerBuffer * pointByteSize);
        const pointDataView = new DataView(pointData);

        let sumX = 0, sumY = 0, sumZ = 0;
        cell.indices.forEach((j, slot) => {
            const posIndex = j * 3;
            const pointOffset = slot * pointByteSize;

            pointDataView.setFloat32(pointOffset,      positions[posIndex],     true);
            pointDataView.setFloat32(pointOffset +  4, positions[posIndex + 1], true);
            pointDataView.setFloat32(pointOffset +  8, positions[posIndex + 2], true);
            pointDataView.setUint32( pointOffset + 12, colors[j],               true);
            // normals (needed by DISKS / BILLBOARDS / GAUSSIANS shaders)
            if (normals) {
                pointDataView.setFloat32(pointOffset + 16, normals[posIndex],     true);
                pointDataView.setFloat32(pointOffset + 20, normals[posIndex + 1], true);
                pointDataView.setFloat32(pointOffset + 24, normals[posIndex + 2], true);
            }
            pointDataView.setFloat32(pointOffset + 28, 0.0, true); // padding

            sumX += positions[posIndex];
            sumY += positions[posIndex + 1];
            sumZ += positions[posIndex + 2];
        });

        const centerX = sumX / count;
        const centerY = sumY / count;
        const centerZ = sumZ / count;

        const pointBuffer = device.createBuffer({
            size: maxPointsPerBuffer * pointByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(pointBuffer, 0, pointData);

        const renderingBindGroup = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: pointBuffer } },
                { binding: 1, resource: { buffer: dummyClassColorsBuffer } },
                { binding: 2, resource: { buffer: useClassColorsBuffer } },
            ],
        });

        pointclouds.push({
            pointBuffer,
            renderingBindGroup,
            numberOfPoints: count,
            center: [centerX, centerY, centerZ],
        });
    }
}


function createDepthRangeBindGroup(minDepth, maxDepth, pipeline = null) {
    // če gre za QUADS/BILLBOARDS/GAUSSIANS, 
    // je pipeline različen od null, sicer pa uporabimo renderingPipeline, 
    const pl = pipeline || renderingPipeline;
    const buf = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([minDepth, maxDepth]));
    return {
        buffer: buf,
        bindGroup: device.createBindGroup({
            layout: pl.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    };
}

function createTargetPositionBindGroup(pos, pipeline) {
    const buf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, new Float32Array([pos[0], pos[1], pos[2], 0.0]));
    return {
        buffer: buf,
        bindGroup: device.createBindGroup({
            layout: pipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
    };
}

function createScreenParamsBindGroup(targetPosition, cameraPosition, pointSize, pipeline) {
    // SceneParams buffer
    const sceneParamsBuffer = device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    
    const sceneData = new Float32Array(12);
    sceneData[0] = targetPosition[0]; sceneData[1] = targetPosition[1]; sceneData[2] = targetPosition[2]; sceneData[3] = 0;
    sceneData[4] = cameraPosition[0]; sceneData[5] = cameraPosition[1]; sceneData[6] = cameraPosition[2]; sceneData[7] = 0;
    sceneData[8] = pointSize; sceneData[9] = 0; sceneData[10] = 0; sceneData[11] = 0;
    device.queue.writeBuffer(sceneParamsBuffer, 0, sceneData);


    const sceneBG = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(3),
        entries: [{ binding: 0, resource: { buffer: sceneParamsBuffer } }],
    });

    return sceneBG;
}

function createMatricesBindGroup(projectionViewMatrix, viewMatrix, pipeline = null) {
    // če gre za QUADS/BILLBOARDS/GAUSSIANS, 
    // je pipeline različen od null, sicer pa uporabimo renderingPipeline, 
    const pl = pipeline || renderingPipeline;
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
            layout: pl.getBindGroupLayout(1),
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
            mode = "POINTS",
            pointSize = 0.01,
        } = params;

        let pipeline;
        if (mode === "POINTS") {
            pipeline = renderingPipeline;
        } else {
            pipeline = quadPipelines[mode];
        }

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

        let outputData;

        // Reconstruction pipeline deluje edino za POINTS tip izrisovanja.
        const effectiveReconstruction = useReconstruction && mode === "POINTS";

        if (effectiveReconstruction) {
            await renderFullScene(projectionViewMatrix, viewMatrix, depthTexture, targetPosition);

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
                    projectionViewMatrix, viewMatrix, depthTexture, composer,
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
            outputData = await capturePointCloudImage(projectionViewMatrix, viewMatrix, depthTexture, targetPosition, cameraPosition, pointSize, mode);
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

// renderFullScene vedno uporabi renderingPipeline (POINTS) za izračun globine.
async function renderFullScene(projectionViewMatrix, viewMatrix, depthTexture, targetPosition) {
    const { bindGroup: matricesBG, mvpBuffer: mvpBuf, viewBuffer: viewBuf } =
        createMatricesBindGroup(projectionViewMatrix, viewMatrix, renderingPipeline);

    const { bindGroup: depthRangeBG, buffer: drBuf } =
        createDepthRangeBindGroup(0, 1e6, renderingPipeline);

    const { bindGroup: targetBG, buffer: tBuf } =
        createTargetPositionBindGroup(targetPosition, renderingPipeline);

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
    renderPass.setBindGroup(1, matricesBG);
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
    mvpBuf.destroy();
    viewBuf.destroy();
    drBuf.destroy();
    tBuf.destroy();
}

// renderPointsInDepthRange vedno uporablja renderingPipeline.
async function renderPointsInDepthRange(
    minDepth, maxDepth, near, far,
    projectionViewMatrix, viewMatrix, depthTexture, composer,
    targetPosition
) {
    const { bindGroup: matricesBG, mvpBuffer: mvpBuf, viewBuffer: viewBuf } =
        createMatricesBindGroup(projectionViewMatrix, viewMatrix, renderingPipeline);

    const { bindGroup: depthRangeBG, buffer: drBuf } =
        createDepthRangeBindGroup(minDepth, maxDepth, renderingPipeline);

    const { bindGroup: targetBG, buffer: tBuf } =
        createTargetPositionBindGroup(targetPosition, renderingPipeline);

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
    renderPass.setBindGroup(1, matricesBG);
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
    mvpBuf.destroy();
    viewBuf.destroy();
    drBuf.destroy();
    tBuf.destroy();
}

async function capturePointCloudImage(projectionViewMatrix, viewMatrix, depthTexture, targetPosition, cameraPosition, pointSize, mode) {
    // GAUSSIANS: najprej sortiramo točke v oblaku glede na njihovo globino
    // v odvisnosti od pogleda kamere, da zagotovimo pravilen back-to-front.
    if (mode === "GAUSSIANS") {
        for (const pc of pointclouds) {
            sortPointCloud(pc, projectionViewMatrix);
        }
        await device.queue.onSubmittedWorkDone();
    }

    // Sortiramo še same kose oblaka (batches) glede na globino njihovega centra, 
    // da zagotovimo pravilno zaporedje risanja med različnimi batchi.
    const batchesWithDepth = pointclouds.map((pc) => ({
        pc,
        depth: getBatchDepth(pc, projectionViewMatrix),
    }));
    batchesWithDepth.sort((a, b) => b.depth - a.depth);

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

    if (mode === "POINTS") {
        const drBuf = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(drBuf, 0, new Float32Array([0, 1e6]));
        const tBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(tBuf, 0, new Float32Array([targetPosition[0], targetPosition[1], targetPosition[2], 0]));

        const mvpBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(mvpBuf, 0, new Float32Array(projectionViewMatrix));
        const viewBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(viewBuf, 0, new Float32Array(viewMatrix));

        const matricesBG = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: mvpBuf } }, { binding: 1, resource: { buffer: viewBuf } }],
        });
        const depthRangeBG = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: drBuf } }],
        });
        const targetBG = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: tBuf } }],
        });

        renderPass.setPipeline(renderingPipeline);
        renderPass.setBindGroup(1, matricesBG);
        renderPass.setBindGroup(2, depthRangeBG);
        renderPass.setBindGroup(3, targetBG);
        for (const { pc } of batchesWithDepth) {
            renderPass.setBindGroup(0, device.createBindGroup({
                layout: renderingPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pc.pointBuffer } },
                    { binding: 1, resource: { buffer: dummyClassColorsBuffer } },
                    { binding: 2, resource: { buffer: useClassColorsBuffer } },
                ],
            }));
            renderPass.draw(pc.numberOfPoints);
        }

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        mvpBuf.destroy(); viewBuf.destroy(); drBuf.destroy(); tBuf.destroy();

    } else {
        const pipeline = quadPipelines[mode];

        const mvpBuf = device.createBuffer({ 
            size: 64, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST 
        });
        device.queue.writeBuffer(mvpBuf, 0, new Float32Array(projectionViewMatrix));
        const viewBuf = device.createBuffer({ 
            size: 64, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST 
        });
        device.queue.writeBuffer(viewBuf, 0, new Float32Array(viewMatrix));

        const drBuf = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(drBuf, 0, new Float32Array([0, 1e6]));

        const sceneBG = createScreenParamsBindGroup(targetPosition, cameraPosition, pointSize, pipeline);

        const matricesBG = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: mvpBuf } }, { binding: 1, resource: { buffer: viewBuf } }],
        });
        const depthRangeBG = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: drBuf } }],
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(1, matricesBG);
        renderPass.setBindGroup(2, depthRangeBG);
        renderPass.setBindGroup(3, sceneBG);
        for (const { pc } of batchesWithDepth) {
            renderPass.setBindGroup(0, device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: pc.pointBuffer } }],
            }));
            renderPass.draw(pc.numberOfPoints * 6);
        }

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        mvpBuf.destroy(); 
        viewBuf.destroy(); 
        drBuf.destroy();
    }

    // Read pixels back to CPU
    const bytesPerPixel = 4;
    const alignedBytesPerRow = Math.ceil((canvas.width * bytesPerPixel) / 256) * 256;
    const outputBuffer = device.createBuffer({
        size: alignedBytesPerRow * canvas.height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const readEncoder = device.createCommandEncoder();
    readEncoder.copyTextureToBuffer(
        { texture: captureTexture, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
        { buffer: outputBuffer, bytesPerRow: alignedBytesPerRow, rowsPerImage: canvas.height },
        [canvas.width, canvas.height, 1]
    );
    device.queue.submit([readEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await outputBuffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(outputBuffer.getMappedRange());
    const tightData = new Uint8Array(canvas.width * canvas.height * bytesPerPixel);
    for (let y = 0; y < canvas.height; y++) {
        tightData.set(
            mapped.subarray(y * alignedBytesPerRow, y * alignedBytesPerRow + canvas.width * bytesPerPixel),
            y * canvas.width * bytesPerPixel
        );
    }

    outputBuffer.unmap();
    outputBuffer.destroy();
    captureTexture.destroy();

    return tightData;
}

// ============================================================================
// SORTING - Bitonic sort
// ============================================================================
// Ustvari buffer z parametri k in j za merge pass
// k - velikost merge bloka, j - korak znotraj merge bloka
const mergeParamCache = new Map();
function getMergeParamBuffer(k, j) {
    const key = (k * 100000 + j); // unikaten integer key
    if (!mergeParamCache.has(key)) {
        const buf = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buf, 0, new Uint32Array([k, j]));
        mergeParamCache.set(key, buf);
    }
    return mergeParamCache.get(key);
}

// Pokliče getMergeParamBuffer za vse potrebne kombinacije k in j (glede na število točk)
function prewarmMergeBuffers(maxN) {
    for (let k = 512; k <= maxN * 2; k *= 2) {
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
            getMergeParamBuffer(k, j);
        }
    }
}

// Vstvari binding groupe za sortiranje - priprava, lokalno sortiranje in globalno sortiranje
function buildSortBindGroups(pointcloud) {
    if (!preparationPipeline || !localSortPipeline || !globalSortPipeline) {
        return;
    }

    const n = pointcloud.numberOfPoints;

    // Reusable view matrix buffer
    pointcloud.prepViewBuffer = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const numBuf = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(numBuf, 0, new Uint32Array([n]));

    // Bind group za pripravo (preparation pass)
    pointcloud.prepBG0 = device.createBindGroup({
        layout: preparationPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: pointcloud.pointBuffer } }],
    });
    pointcloud.prepBG1 = device.createBindGroup({
        layout: preparationPipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: { buffer: pointcloud.prepViewBuffer } },
            { binding: 1, resource: { buffer: numBuf } },
        ],
    });

    // Bind group za local sort
    pointcloud.localSortBG = device.createBindGroup({
        layout: localSortPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointcloud.pointBuffer } },
            { binding: 1, resource: { buffer: numBuf } },
        ],
    });

    // Bind groupi za global sort - potrebujemo jih več, glede na različne kombinacije k in j
    pointcloud.globalSortBGs = new Map();
    for (let k = 512; k <= n * 2; k *= 2) {
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
            const key = k * 100000 + j;
            pointcloud.globalSortBGs.set(key, device.createBindGroup({
                layout: globalSortPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pointcloud.pointBuffer } },
                    { binding: 1, resource: { buffer: numBuf } },
                    { binding: 2, resource: { buffer: getMergeParamBuffer(k, j) } },
                ],
            }));
        }
    }
}

function sortPointCloud(pointcloud, viewMatrix) {
    if (!preparationPipeline || !localSortPipeline || !globalSortPipeline) {
        console.warn("⚠️ Sorting pipelines not available, skipping sort");
        return;
    }

    const n = pointcloud.numberOfPoints;
    const numWorkgroups = Math.ceil(n / 256);

    console.log(`sorting: numberOfPoints=${pointcloud.numberOfPoints}, paddedSize=${n}, ratio=${n/pointcloud.numberOfPoints}`);


    device.queue.writeBuffer(pointcloud.prepViewBuffer, 0, new Float32Array(viewMatrix));

    // Preparation pass
    // izračunamo globino vsake točke glede na trenutni pogled kamere in shranimo v buffer
    // To je sort key za kasnejše sortiranje
    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(preparationPipeline);
    pass.setBindGroup(0, pointcloud.prepBG0);
    pass.setBindGroup(1, pointcloud.prepBG1);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Local sort pass - sortiramo točke znotraj vsake skupine
    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(localSortPipeline);
    pass.setBindGroup(0, pointcloud.localSortBG);
    pass.dispatchWorkgroups(numWorkgroups);
    pass.end();
    device.queue.submit([encoder.finish()]);

    // Global sort pass - sortiramo skupine med seboj (bitonic sort)
    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(globalSortPipeline);
    for (let k = 512; k <= n * 2; k *= 2) {
        for (let j = k / 2; j >= 1; j = Math.floor(j / 2)) {
            const key = k * 100000 + j;
            pass.setBindGroup(0, pointcloud.globalSortBGs.get(key));
            pass.dispatchWorkgroups(numWorkgroups);
        }
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
}

// Pomožna funkcija za izračun globine batcha glede na center celice in trenutni pogled kamere
function getBatchDepth(pointcloud, projectionViewMatrix) {
    const [px, py, pz] = pointcloud.center;
    const m = projectionViewMatrix;
    const clipZ = m[2]*px + m[6]*py + m[10]*pz + m[14];
    const clipW = m[3]*px + m[7]*py + m[11]*pz + m[15];
    return clipZ / clipW;
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