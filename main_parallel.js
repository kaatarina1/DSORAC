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
    workerCount: Math.min(8, navigator.hardwareConcurrency || 6),
    hemisphereRadius: 0.75,
    hemisphereSamples: 16,
    lasFile: "./data/cropped_filtered_1.las",
    targetPositions: [
        [0, -0.17, 0],
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

// Camera controls
let cameraPosition = [0, 0, 0];
let cameraTarget = [0, -0.17, 0];
let yaw = 0;
let pitch = 0;
let distance = 0.2;

const pointerController = new PointerController();

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
    console.log("🚀 Starting parallel generation with batching...");

    try {
        // Najprej generiramo vse pozicije kamer in naloge, preden začnemo z delom workerjev
        const allTasks = [];
        let imageIndex = 0;

        for (const target of CONFIG.targetPositions) {
            const poses = camPositionHelper.generateFibonacciHemisphereAroundCamera(
                CONFIG.hemisphereSamples,
                CONFIG.hemisphereRadius,
                target
            );
            console.log(`Generated ${poses.length} camera poses around target ${target}`);

            for (const pos of poses) {
                const dist = Math.sqrt(
                    Math.pow(pos[0] - target[0], 2) +
                    Math.pow(pos[1] - target[1], 2) +
                    Math.pow(pos[2] - target[2], 2)
                );

                if (dist < 0.1) continue;

                allTasks.push({
                    cameraPosition: pos,
                    target: target,
                    imageIndex: imageIndex++,
                    targetPosition: CONFIG.targetPositions[0]
                });
            }
        }

        console.log(`📋 Total images to generate: ${allTasks.length}`);
        
        // Initializiramo ZIP arhiv za shranjevanje rezultatov
        zip = new JSZip();
        
        const BATCH_SIZE = 20;
        const allResults = [];
        const numBatches = Math.ceil(allTasks.length / BATCH_SIZE);

        for (let batch = 0; batch < numBatches; batch++) {
            const start = batch * BATCH_SIZE;
            const end = Math.min(start + BATCH_SIZE, allTasks.length);
            const batchTasks = allTasks.slice(start, end);
            
            console.log(`\n📦 Processing batch ${batch + 1}/${numBatches} (images ${start + 1}-${end})...`);
            
            // Ustvarimo nov WorkerPool za vsak batch, da zagotovimo popolno sprostitev GPU virov po vsakem batchu
            workerPool = new WorkerPool(CONFIG.workerCount);
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
            
            console.log(`⏳ Waiting for GPU and message queue to clear...`);
            await new Promise(resolve => setTimeout(resolve, 4000));
            
            // Ustavimo workerje, da sprostimo GPU vire
            await workerPool.shutdown();
            console.log(`✓ Batch ${batch + 1} complete, workers shut down`);
            
            // Dodamo majhen zamik pred začetkom naslednjega batcha, da zagotovimo, da so vsi GPU viri sproščeni
            if (batch < numBatches - 1) {
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
🚀 Workers: ${CONFIG.workerCount}
📦 Batches: ${numBatches} (20 images each)
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

console.log(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 READY - Batch Processing Mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Workers: ${CONFIG.workerCount}
Batch size: 20 images (auto-restart)
Press 'T' to start
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);