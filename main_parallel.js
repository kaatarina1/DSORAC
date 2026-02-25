import * as mat4 from "./js/mat4.js";
import { PointerController } from "./js/PointerController.js";
import { LasLoader } from "./js/LasLoader.js";
import { CameraPosition } from "./js/CameraPosition.js";
import JSZip from "jszip";
import saveAs from "file-saver";

import WorkerScript from './imageGenerationWorker.js?worker';

const CONFIG = {
    canvasWidth: 1024,
    canvasHeight: 512,
    workerCount: Math.min(7, navigator.hardwareConcurrency || 6),
    batchSize: 16,
    reconstructionColorOrder: "rgb",
    simpleRenderingColorOrder: "bgr",
    hemisphereRadii: [0.17/*, 0.35, 0.95*/],
    imagesPerCombination: 7,
    lasFile: "./data/cropped_filtered_1.las",
    targetPositions: [
        [0, -0.17, 0],
        // [0.1, -0.17, 0],
        // [-0.1, -0.17, 0],
        // [0, -0.17, 0.1],
        // [0, -0.17, -0.1],
        // [0.1, -0.17, 0.1],
        // [-0.1, -0.17, 0.1],
        // [0.1, -0.17, -0.1],
        // [-0.1, -0.17, -0.1],
    ]
};

console.log(`Using ${CONFIG.workerCount} workers`);

class WorkerPool {
    constructor(workerCount) {
        this.workerCount = workerCount;
        this.workers = [];
        this.taskQueue = [];
        this.initialized = false;
    }

    async initialize(config) {
        console.log(`Initializing ${this.workerCount} workers...`);
        
        for (let i = 0; i < this.workerCount; i++) {
            console.log(`Creating worker ${i}...`);
            
            const worker = new WorkerScript();
            
            const initPromise = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Worker ${i} timeout`));
                }, 60000);

                worker.onmessage = (e) => {
                    if (e.data.type === 'INIT_COMPLETE') {
                        clearTimeout(timeout);
                        console.log(`✓ Worker ${i} ready`);
                        resolve();
                    } else if (e.data.type === 'ERROR') {
                        clearTimeout(timeout);
                        reject(new Error(e.data.error));
                    }
                };

                worker.onerror = (error) => {
                    clearTimeout(timeout);
                    console.error(`Worker ${i} creation error:`, error);
                    reject(error);
                };
            });

            worker.postMessage({
                type: 'INIT',
                data: {
                    width: config.width,
                    height: config.height,
                }
            });

            this.workers.push({ worker, id: i, busy: false });
            
            try {
                await initPromise;
            } catch (error) {
                console.error(`Failed to initialize worker ${i}:`, error);
                this.workers.forEach(({ worker }) => worker.terminate());
                this.workers = [];
                throw error;
            }
        }
        
        this.initialized = true;
        console.log(`✓ All ${this.workerCount} workers initialized`);
    }

    async processImages(tasks, progressCallback) {
        if (!this.initialized) {
            throw new Error('WorkerPool not initialized');
        }

        return new Promise((resolve, reject) => {
            const results = [];
            const processedImages = new Set();
            const failedImages = new Set();
            let completed = 0;
            let failed = 0;
            const totalTasks = tasks.length;
            let resolveTimer = null;

            this.workers.forEach(({ worker, id }) => {
                worker.onmessage = (e) => {
                    switch (e.data.type) {
                        case 'IMAGE_COMPLETE':
                            // Preveri, ali je ta slika že bila obdelana (v primeru podvajanja sporočil)
                            if (processedImages.has(e.data.imageIndex)) {
                                console.warn(`⚠️ Duplicate completion for image ${e.data.imageIndex}`);
                                break;
                            }
                            
                            processedImages.add(e.data.imageIndex);
                            completed++;
                            results.push({
                                imageIndex: e.data.imageIndex,
                                blob: e.data.blob,
                                metadata: e.data.metadata
                            });

                            if (progressCallback) {
                                progressCallback({
                                    completed,
                                    total: totalTasks,
                                    percentage: (completed / totalTasks) * 100,
                                    workerId: id
                                });
                            }

                            this.assignNextTask(id);
                            break;

                        case 'PROGRESS':
                            if (e.data.message) {
                                console.log(`Worker ${id}: ${e.data.message}`);
                            }
                            break;

                        case 'ERROR':
                            const errorImageIndex = e.data.imageIndex;
                            if (!failedImages.has(errorImageIndex) && !processedImages.has(errorImageIndex)) {
                                failedImages.add(errorImageIndex);
                                failed++;
                                console.error(`Worker ${id} error on image ${errorImageIndex}:`, e.data.error);
                            }
                            // Ne kličemo assignNextTask tukaj, da preprečimo nadaljnje nalaganje nalog, če pride do napake
                            break;
                    }

                    // Preveri, ali so vse naloge zaključene (upoštevajoč tako uspešne kot neuspešne)
                    if (completed + failed >= totalTasks) {
                        if (resolveTimer) clearTimeout(resolveTimer);
                        resolveTimer = setTimeout(() => {
                            if (failed > 0) {
                                console.warn(`⚠️ Batch completed: ${completed} succeeded, ${failed} failed`);
                                console.warn(`Failed images:`, Array.from(failedImages));
                            }
                            resolve(results);
                        }, 500);
                    }
                };

                worker.onerror = (error) => {
                    console.error(`Worker ${id} fatal error:`, error);
                    failed++;
                    if (completed + failed >= totalTasks) {
                        if (resolveTimer) clearTimeout(resolveTimer);
                        resolveTimer = setTimeout(() => reject(error), 500);
                    }
                };
            });

            this.taskQueue = [...tasks];
            
            // Dodamo majhen zamik pri dodeljevanju nalog, da zmanjšamo začetno obremenitev GPU
            this.workers.forEach(({ id }, index) => {
                setTimeout(() => this.assignNextTask(id), index * 200);
            });
        });
    }

    assignNextTask(workerId) {
        const workerInfo = this.workers.find(w => w.id === workerId);
        
        if (this.taskQueue.length === 0) {
            workerInfo.busy = false;
            return;
        }

        const task = this.taskQueue.shift();
        workerInfo.busy = true;
        
        // Dodamo majhen zamik pri začetku novih nalog, da zmanjšamo obremenitev GPU
        setTimeout(() => {
            workerInfo.worker.postMessage({
                type: 'GENERATE_IMAGE',
                data: task
            });
        }, 100);
    }

    async shutdown() {
        console.log('Shutting down workers...');
        
        // Pošljemo sporočilo za zaustavitev vsem workerjem, da lahko počistijo GPU vire
        for (const { worker } of this.workers) {
            worker.postMessage({ type: 'SHUTDOWN' });
        }
        
        // Počakamo, da workerji obdelajo sporočilo za zaustavitev
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // Ustavimo vse workerje, da zagotovimo sprostitev vseh virov
        for (const { worker } of this.workers) {
            worker.terminate();
        }
        
        this.workers = [];
        this.initialized = false;
        console.log('✓ All workers terminated');
    }
}

// ============================================================================
// MAIN SETUP
// ============================================================================

const adapter = await navigator.gpu.requestAdapter();
const hasBGRA8unormStorage = adapter.features.has("bgra8unorm-storage");
const device = await adapter?.requestDevice({
    requiredFeatures: hasBGRA8unormStorage 
        ? ["bgra8unorm-storage", "float32-filterable", "float32-blendable", "timestamp-query"] 
        : ["float32-filterable", "float32-blendable", "timestamp-query"],
});

const canvas = document.querySelector("canvas");
const reconstructionToggle = document.getElementById("reconstruction-toggle");
canvas.width = CONFIG.canvasWidth;
canvas.height = CONFIG.canvasHeight;

const context = canvas.getContext("webgpu");
const format = hasBGRA8unormStorage
    ? navigator.gpu.getPreferredCanvasFormat()
    : "rgba8unorm";
context.configure({ device, format, alphaMode: "premultiplied" });

console.log("Loading point cloud...");
const lasLoader = new LasLoader(CONFIG.lasFile);
const lasData = await lasLoader.loadLasData();
const positions = lasData.positions;
const colors = lasData.colors;
const colorsRGB = lasData.colorsRGB;
const scaleFactor = lasData.scaleFactor;

console.log(`Loaded ${positions.length / 3} points`);

// Izračunamo bounding box oblaka točk
let bbMin = [Infinity, Infinity, Infinity];
let bbMax = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < positions.length; i += 3) {
    bbMin[0] = Math.min(bbMin[0], positions[i]);
    bbMin[1] = Math.min(bbMin[1], positions[i + 1]);
    bbMin[2] = Math.min(bbMin[2], positions[i + 2]);
    bbMax[0] = Math.max(bbMax[0], positions[i]);
    bbMax[1] = Math.max(bbMax[1], positions[i + 1]);
    bbMax[2] = Math.max(bbMax[2], positions[i + 2]);
}
console.log("Point cloud bounding box:", bbMin, bbMax);

// Naložimo shaderje in ustvarimo render pipeline
const renderingCode = await fetch("./shaders/rendering.wgsl").then(r => r.text());
const renderingModule = device.createShaderModule({ code: renderingCode });

const renderingPipeline = device.createRenderPipeline({
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

const camPositionHelper = new CameraPosition(canvas, device, renderingPipeline);

// Matrix buffers: MVP (binding 0) + view matrix (binding 1) in group 1
const mvpBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const viewMatrixBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const matricesBindGroup = device.createBindGroup({
    layout: renderingPipeline.getBindGroupLayout(1),
    entries: [
        { binding: 0, resource: { buffer: mvpBuffer } },
        { binding: 1, resource: { buffer: viewMatrixBuffer } },
    ],
});

// Camera controls
let cameraPosition = [0, 0, 0];
let cameraTarget = [0, -0.17, 0];
let yaw = 0;
let pitch = 0;
let distance = 0.2;

const pointerController = new PointerController(canvas);

pointerController.addEventListener("pointermove", (e) => {
    const dx = e.movementX;
    const dy = e.movementY;
    const sensitivity = 0.005;
    
    if (e.buttons === 1) {
        yaw -= dx * sensitivity;
        pitch -= dy * sensitivity;
        pitch = Math.max(-Math.PI + 0.1, Math.min(-0.1, pitch));
        updateCameraOrbit();
    }
});

function updateCameraOrbit() {
    cameraPosition[0] = cameraTarget[0] + distance * Math.cos(pitch) * Math.sin(yaw);
    cameraPosition[1] = cameraTarget[1] + distance * Math.sin(pitch);
    cameraPosition[2] = cameraTarget[2] + distance * Math.cos(pitch) * Math.cos(yaw);
}

updateCameraOrbit();

document.addEventListener("keydown", (event) => {
    const moveSpeed = 0.01;
    const forward = mat4.normalize([
        cameraTarget[0] - cameraPosition[0],
        0,
        cameraTarget[2] - cameraPosition[2],
    ]);
    const right = mat4.cross(forward, [0, 1, 0]);
    
    switch(event.key) {
        case "ArrowLeft":
            cameraPosition[0] -= right[0] * moveSpeed;
            cameraPosition[2] -= right[2] * moveSpeed;
            cameraTarget[0] -= right[0] * moveSpeed;
            cameraTarget[2] -= right[2] * moveSpeed;
            break;
        case "ArrowRight":
            cameraPosition[0] += right[0] * moveSpeed;
            cameraPosition[2] += right[2] * moveSpeed;
            cameraTarget[0] += right[0] * moveSpeed;
            cameraTarget[2] += right[2] * moveSpeed;
            break;
        case "ArrowUp":
            cameraPosition[0] += forward[0] * moveSpeed;
            cameraPosition[2] += forward[2] * moveSpeed;
            cameraTarget[0] += forward[0] * moveSpeed;
            cameraTarget[2] += forward[2] * moveSpeed;
            break;
        case "ArrowDown":
            cameraPosition[0] -= forward[0] * moveSpeed;
            cameraPosition[2] -= forward[2] * moveSpeed;
            cameraTarget[0] -= forward[0] * moveSpeed;
            cameraTarget[2] -= forward[2] * moveSpeed;
            break;
        case "PageUp":
            cameraPosition[1] += moveSpeed;
            cameraTarget[1] += moveSpeed;
            break;
        case "PageDown":
            cameraPosition[1] -= moveSpeed;
            cameraTarget[1] -= moveSpeed;
            break;
    }
});

// ============================================================================
// PARALLEL GENERATION WITH BATCHING
// ============================================================================

let workerPool = null;
let zip = null;

document.addEventListener("keydown", async (event) => {
    if (event.key === "T" || event.key === "t") {
        await generateImagesParallel();
    }
});

async function generateImagesParallel() {
    const startTime = performance.now();
    const useReconstruction = reconstructionToggle ? reconstructionToggle.checked : true;
    const useParallelBatching = useReconstruction;
    const activeWorkerCount = useParallelBatching ? CONFIG.workerCount : 1;
    const activeBatchSize = useParallelBatching ? CONFIG.batchSize : Number.MAX_SAFE_INTEGER;
    const colorOrder = useReconstruction
        ? CONFIG.reconstructionColorOrder
        : CONFIG.simpleRenderingColorOrder;

    console.log(
        `🚀 Starting generation (${useParallelBatching ? "parallel + batched" : "single worker, no batching"})...`
    );
    console.log(`Capture mode: ${useReconstruction ? "reconstruction" : "point cloud only"}`);
    console.log(`Output color order: ${colorOrder.toUpperCase()}`);

    try {
        // Najprej generiramo vse pozicije kamer in naloge, preden začnemo z delom workerjev
        const allTasks = [];
        let imageIndex = 0;

        for (const target of CONFIG.targetPositions) {
            for (const radius of CONFIG.hemisphereRadii) {
                const oversampledCount = Math.max(
                    CONFIG.imagesPerCombination * 2,
                    CONFIG.imagesPerCombination + 1
                );

                const poses = camPositionHelper.generateFibonacciHemisphereAroundCamera(
                    oversampledCount,
                    radius,
                    target
                );

                const selectedPoses = poses.slice(0, CONFIG.imagesPerCombination);

                console.log(
                    `Generated ${selectedPoses.length}/${CONFIG.imagesPerCombination} camera poses around target ${target} at radius ${radius}`
                );

                for (const pos of selectedPoses) {
                    allTasks.push({
                        cameraPosition: pos,
                        target: target,
                        imageIndex: imageIndex++,
                        targetPosition: target,
                        useReconstruction,
                        colorOrder,
                        radius
                    });
                }
            }
        }

        console.log(`📋 Total images to generate: ${allTasks.length}`);
        
        // Initializiramo ZIP arhiv za shranjevanje rezultatov
        zip = new JSZip();

        const allResults = [];
        const numBatches = Math.ceil(allTasks.length / activeBatchSize);

        for (let batch = 0; batch < numBatches; batch++) {
            const start = batch * activeBatchSize;
            const end = Math.min(start + activeBatchSize, allTasks.length);
            const batchTasks = allTasks.slice(start, end);
            
            console.log(`\n📦 Processing batch ${batch + 1}/${numBatches} (images ${start + 1}-${end})...`);
            
            // Ustvarimo nov WorkerPool za vsak batch, da zagotovimo popolno sprostitev GPU virov po vsakem batchu
            workerPool = new WorkerPool(activeWorkerCount);
            await workerPool.initialize({
                width: CONFIG.canvasWidth,
                height: CONFIG.canvasHeight,
            });

            // Obdelamo trenutni batch nalog in spremljamo napredek
            const batchResults = await workerPool.processImages(
                batchTasks,
                (progress) => {
                    const totalCompleted = start + progress.completed;
                    const elapsed = (performance.now() - startTime) / 1000 / 60;
                    const remaining = (elapsed / totalCompleted) * (allTasks.length - totalCompleted);
                    
                    console.log(
                        `✓ ${totalCompleted}/${allTasks.length} ` +
                        `(${((totalCompleted / allTasks.length) * 100).toFixed(1)}%) | ` +
                        `Batch: ${progress.completed}/${batchTasks.length} | ` +
                        `Elapsed: ${elapsed.toFixed(1)}m | ` +
                        `Remaining: ~${remaining.toFixed(1)}m`
                    );
                }
            );

            allResults.push(...batchResults);
            
            console.log(`📊 Batch ${batch + 1} results: ${batchResults.length}/${batchTasks.length} images received`);

            if (useParallelBatching) {
                console.log(`⏳ Waiting for GPU and message queue to clear...`);
                await new Promise(resolve => setTimeout(resolve, 4000));
            }
            
            // Ustavimo workerje, da sprostimo GPU vire
            await workerPool.shutdown();
            console.log(`✓ Batch ${batch + 1} complete, workers shut down`);
            
            // Dodamo majhen zamik pred začetkom naslednjega batcha, da zagotovimo, da so vsi GPU viri sproščeni
            if (useParallelBatching && batch < numBatches - 1) {
                console.log(`⏳ Cooling down GPU before next batch...`);
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }


        // Dodamo vse slike v ZIP arhiv in pripravimo podatke za izvoz
        console.log(`\n📦 Adding images to ZIP...`);
        let addedCount = 0;
        const receivedIndices = new Set();
        
        for (const result of allResults) {
            if (result && result.blob && result.metadata) {
                receivedIndices.add(result.imageIndex);
                zip.file(`images/${result.metadata.filename}`, result.blob);
                addedCount++;
            }
        }
        
        console.log(`✓ Added ${addedCount}/${allResults.length} images to ZIP`);
        
        // Preverimo ali obstajajo manjkajoče slike
        const expectedIndices = Array.from({length: allTasks.length}, (_, i) => i);
        const missingIndices = expectedIndices.filter(i => !receivedIndices.has(i));
        if (missingIndices.length > 0) {
            console.warn(`⚠️ Missing ${missingIndices.length} images:`, missingIndices.slice(0, 20));
        }
        
        const successfulResults = allResults.filter(r => r && r.metadata);        
        console.log(`✓ Successfully generated ${successfulResults.length}/${allTasks.length} images`);

        // Izvozimo rezultate (cameras.txt, images.txt, points3D.txt) v ZIP arhiv
        await exportResults(
            successfulResults
                .map(r => ({
                    imageId: r.imageIndex + 1,
                    filename: r.metadata.filename,
                    quat: r.metadata.quat,
                    t: r.metadata.t,
                    camera_id: 1,
                    name: r.metadata.filename,
                    fx: r.metadata.fx,
                    fy: r.metadata.fy,
                    cx: r.metadata.cx,
                    cy: r.metadata.cy,
                    width: r.metadata.width,
                    height: r.metadata.height,
                }))
        );

        const totalTime = (performance.now() - startTime) / 1000 / 60;

        console.log(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ PARALLEL GENERATION COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Images: ${successfulResults.length}/${allTasks.length}
⏱️  Time: ${totalTime.toFixed(1)} minutes
⚡ Avg: ${successfulResults.length > 0 ? (totalTime * 60 / successfulResults.length).toFixed(1) : 'N/A'}s/image
🚀 Workers: ${activeWorkerCount}
📦 Batches: ${numBatches} (${useParallelBatching ? activeBatchSize : allTasks.length} images each)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        `);

    } catch (error) {
        console.error("❌ Error:", error);
    } finally {
        if (workerPool) {
            await workerPool.shutdown();
        }
    }
}

async function exportResults(capturedData) {
    // Preverimo ali je ZIP objekt veljaven pred uporabo
    console.log("Exporting results to ZIP...");
    if (!zip) {
        console.error('❌ ZIP object is null - cannot export');
        return;
    }
    
    zip.file("cameras.txt", camPositionHelper.writeCamerasTxt(capturedData));
    zip.file("images.txt", camPositionHelper.writeImagesTxt(capturedData, scaleFactor));
    zip.file("points3D.txt", camPositionHelper.writePoints3DTxt(positions, colorsRGB, scaleFactor));

    const blob = await zip.generateAsync({ type: "blob" });
    saveAs(blob, "export_bundle.zip");
    console.log("✓ Export saved");
}

// Interactive preview render loop
const pointByteSize = 16;
const maxNumberOfPointsPerBuffer = 1024 * 1024;
const numberOfAllPoints = positions.length / 3;
const pointclouds = [];

for (let i = 0; i < Math.ceil(numberOfAllPoints / maxNumberOfPointsPerBuffer); i++) {
    const startIndex = i * maxNumberOfPointsPerBuffer;
    const numberOfPoints = Math.min(maxNumberOfPointsPerBuffer, numberOfAllPoints - startIndex);

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

    pointclouds.push({ pointBuffer, renderingBindGroup, numberOfPoints });
}

let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: "depth32float",
});

let lastSize = { width: null, height: null };

function frame() {
    const size = canvas.getBoundingClientRect();
    if (size.width !== lastSize.width || size.height !== lastSize.height) {
        canvas.width = lastSize.width = size.width;
        canvas.height = lastSize.height = size.height;
        depthTexture.destroy();
        depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
            format: "depth32float",
        });
    }

    // Compute matrices with tight near/far from bounding box
    const { viewMatrix, projectionViewMatrix, near, far } =
        camPositionHelper.computeCameraMatrix(cameraPosition, cameraTarget, canvas, bbMin, bbMax);

    device.queue.writeBuffer(mvpBuffer, 0, new Float32Array(projectionViewMatrix));
    device.queue.writeBuffer(viewMatrixBuffer, 0, new Float32Array(viewMatrix));

    // Depth range: pass huge range so nothing is clipped in preview
    const depthRangeBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(depthRangeBuffer, 0, new Float32Array([0, 1e6]));
    const depthRangeBindGroup = device.createBindGroup({
        layout: renderingPipeline.getBindGroupLayout(2),
        entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
    });

    // Target position — vec4f padded for alignment
    const targetPosBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const tp = CONFIG.targetPositions[0];
    device.queue.writeBuffer(targetPosBuf, 0, new Float32Array([tp[0], tp[1], tp[2], 0.0]));
    const targetPosBindGroup = device.createBindGroup({
        layout: renderingPipeline.getBindGroupLayout(3),
        entries: [{ binding: 0, resource: { buffer: targetPosBuf } }],
    });

    const commandEncoder = device.createCommandEncoder();
    const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
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
    renderPass.setBindGroup(2, depthRangeBindGroup);
    renderPass.setBindGroup(3, targetPosBindGroup);

    for (const pointcloud of pointclouds) {
        renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
        renderPass.draw(pointcloud.numberOfPoints);
    }
    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

console.log(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 READY - Batch Processing Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Workers: ${CONFIG.workerCount}
Batch size: 20 images (auto-restart)
Press 'T' to start
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);