import * as mat4 from "./js/mat4.js";
import { PointerController } from "./js/PointerController.js";
import { LasLoader } from "./js/LasLoader.js";
import { CameraPosition } from "./js/CameraPosition.js";
import { RenderingControls } from "./js/RenderingControls.js";
import { AdaptiveGrid } from "./js/AdaptiveGrid.js";
import JSZip from "jszip";
import saveAs from "file-saver";

import WorkerScript from './imageGenerationWorker.js?worker';

const CONFIG = {
    canvasWidth: 1024,
    canvasHeight: 512,
    workerCount: Math.min(7, navigator.hardwareConcurrency || 6),
    batchSize: 20,
    reconstructionColorOrder: "rgb",
    simpleRenderingColorOrder: "bgr",
    hemisphereRadii: [/*325, 200, 140, 110*/, 1],
    imagesPerCombination: 110,
    lasFile: "./data/pc/room_normals.las",
    targetPositions: [
        [0, 0, 0],
        // [-40, 15, -20],
        // [40, 15, -20],
        // [-40, 15, 10],
        // [40, 15, 10],
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
                    mode: currentMode,
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
const normals = lasData.normals;
const classifications = lasData.classifications;

function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
    const m = l - c / 2;
    let r = 0, g = 0, b = 0;
    const sec = Math.floor(h * 6) % 6;
    if      (sec === 0) { r = c; g = x; }
    else if (sec === 1) { r = x; g = c; }
    else if (sec === 2) {        g = c; b = x; }
    else if (sec === 3) {        g = x; b = c; }
    else if (sec === 4) { r = x;        b = c; }
    else                { r = c;        b = x; }
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

const classPalette = new Uint32Array(256);
classPalette[0] = 0xFF808080; // class 0 (unclassified) = grey
const golden = 0.6180339887;
for (let ci = 1; ci < 256; ci++) {
    const [r, g, b] = hslToRgb((ci * golden) % 1.0, 0.75, 0.55);
    classPalette[ci] = (r | (g << 8) | (b << 16) | (0xFF << 24)) >>> 0;
}
const classColors = new Uint32Array(positions.length / 3);
for (let i = 0; i < classColors.length; i++) {
    classColors[i] = classPalette[classifications[i]];
}

// Log class -> color mapping (only classes that actually appear in the data)
const presentClasses = new Map();
for (const cls of classifications) {
    if (!presentClasses.has(cls)) {
        const packed = classPalette[cls];
        const r = (packed >>  0) & 0xFF;
        const g = (packed >>  8) & 0xFF;
        const b = (packed >> 16) & 0xFF;
        presentClasses.set(cls, `rgb(${r}, ${g}, ${b})`);
    }
}
console.log(`Class → color mapping (${presentClasses.size} classes):`);
for (const [cls, color] of [...presentClasses].sort((a, b) => a[0] - b[0])) {
    console.log(`  %c  %c class ${cls} → ${color}`, `background:${color};padding:0 8px`, 'background:none');
}

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

// Prilagodimo rendering pipeline za quad/billboard/gaussian
async function loadQuadPipeline(shaderFile, enableBlend) {
    const code = await fetch(shaderFile).then(r => r.text());
    const module = device.createShaderModule({ code });
    const blendState = {
        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    return device.createRenderPipeline({
        vertex: { module },
        fragment: { module, targets: [{ format, ...(enableBlend ? { blend: blendState } : {}) }] }, // blend dodamo samo če gre za gaussian
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: {
            depthWriteEnabled: !enableBlend,  // false za GAUSSIANS, saj pri gaussian za pravilen izris moremo sortirat 
            depthCompare: "less",
            format: "depth32float"
        },
        layout: "auto",
    });
}

const quadPipelines = {
    DISKS:      await loadQuadPipeline("./shaders/rendering_disks.wgsl",      false),
    BILLBOARDS: await loadQuadPipeline("./shaders/rendering_billboards.wgsl", false),
    GAUSSIANS:  await loadQuadPipeline("./shaders/rendering_gaussians.wgsl",  true),
}; // določimo kateri shader uporabljamo

// Render mode state
let currentMode = "POINTS";
let currentPointSize = 0.2;

// SceneParams buffer
const sceneParamsBuffer = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

function writeSceneParams(tp, camPos, pointSize) {
    const data = new Float32Array(12);
    data[0] = tp[0];   data[1] = tp[1];   data[2] = tp[2];   data[3] = 0;   // targetPosition
    data[4] = camPos[0]; data[5] = camPos[1]; data[6] = camPos[2]; data[7] = 0; // cameraPos
    data[8] = pointSize; data[9] = 0; data[10] = 0; data[11] = 0;             // pointSize + pad
    device.queue.writeBuffer(sceneParamsBuffer, 0, data);
}

// Barva ob upodabljanju razredov po segmentaciji
let showClassColors = false;
const classUniformBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(classUniformBuffer, 0, new Uint32Array([0, 0, 0, 0]));

// ============================================================================
// RENDERING CONTROLS UI
// ============================================================================
let useReconstruction = true; // State variable for reconstruction mode

// ── Ortho top-down camera state ──────────────────────────────────────────────
let orthoMode  = false;
const sceneHeight = bbMax[1] - bbMin[1];
const sceneCX  = (bbMin[0] + bbMax[0]) / 2;
const sceneCZ  = (bbMin[2] + bbMax[2]) / 2;
let orthoEyeY  = bbMax[1] + sceneHeight * 0.8;
let orthoPanX  = sceneCX;
let orthoPanZ  = sceneCZ;
let orthoZoom  = Math.max(
    (bbMax[0] - bbMin[0]) / 2,
    (bbMax[2] - bbMin[2]) / 2
) * 1.15;
// ─────────────────────────────────────────────────────────────────────────────

const controls = new RenderingControls({
    modes: ["POINTS", "DISKS", "BILLBOARDS", "GAUSSIANS"],
    currentMode,
    currentPointSize,
    useReconstruction,
    onModeChange: (mode) => { currentMode = mode; },
    onSizeChange: (size) => { currentPointSize = size; },
    onReconstructionChange: (value) => { useReconstruction = value; },
    onOrthoChange: (value) => { orthoMode = value; },
    onClassColorChange: (value) => {
        showClassColors = value;
        device.queue.writeBuffer(classUniformBuffer, 0, new Uint32Array([value ? 1 : 0, 0, 0, 0]));
    },
});
controls.mount(document.body);

// Sorting pipelines
const preparationCode  = await fetch("./shaders/preparation.wgsl").then(r => r.text());
const localSortCode    = await fetch("./shaders/localSort.wgsl").then(r => r.text());
const globalSortCode   = await fetch("./shaders/globalSort.wgsl").then(r => r.text());

const preparationPipeline = device.createComputePipeline({
    compute: { module: device.createShaderModule({ code: preparationCode }), entryPoint: "main" },
    layout: "auto",
});
const localSortPipeline = device.createComputePipeline({
    compute: { module: device.createShaderModule({ code: localSortCode }), entryPoint: "compute" },
    layout: "auto",
});
const globalSortPipeline = device.createComputePipeline({
    compute: { module: device.createShaderModule({ code: globalSortCode }), entryPoint: "mergePass" },
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
let cameraPosition = [0, 20, 0];
let cameraTarget = [0, 0, 0];
let yaw = 0;
let pitch = Math.PI / 2 - 0.1;  // Start looking down from above
let distance = 20;

const pointerController = new PointerController(canvas);

pointerController.addEventListener("pointermove", (e) => {
    const dx = e.movementX;
    const dy = e.movementY;
    const sensitivity = 0.005;
    
    if (orthoMode) {
        const aspect = canvas.width / canvas.height;
        let halfW, halfH;
        if (canvas.width <= canvas.height) {
            halfW = orthoZoom;
            halfH = orthoZoom / aspect;
        } else {
            halfH = orthoZoom;
            halfW = orthoZoom * aspect;
        }
        const worldPerPixelX = (2 * halfW) / canvas.width;
        const worldPerPixelZ = (2 * halfH) / canvas.height;
        orthoPanX -= dx * worldPerPixelX;
        orthoPanZ -= dy * worldPerPixelZ;
    } else {
        yaw -= dx * sensitivity;
        pitch -= dy * sensitivity;
        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2 - 0.1, pitch));
        updateCameraOrbit();
    }
});

let SORT = false;
function updateCameraOrbit() {
    cameraPosition[0] = cameraTarget[0] + distance * Math.cos(pitch) * Math.sin(yaw);
    cameraPosition[1] = cameraTarget[1] + distance * Math.sin(pitch);
    cameraPosition[2] = cameraTarget[2] + distance * Math.cos(pitch) * Math.cos(yaw);
    SORT = true;
}

updateCameraOrbit();

document.addEventListener("keydown", (event) => {
     if (orthoMode) {
        const panStep    = orthoZoom * 0.05;
        const heightStep = sceneHeight * 0.05;
        switch (event.key) {
            case "ArrowLeft":  orthoPanX -= panStep; break;
            case "ArrowRight": orthoPanX += panStep; break;
            case "ArrowUp":    orthoPanZ -= panStep; break;
            case "ArrowDown":  orthoPanZ += panStep; break;
            case "w": case "W":
                orthoEyeY += heightStep;
                console.log(`Ortho eye Y: ${orthoEyeY.toFixed(2)}`);
                break;
            case "s": case "S":
                orthoEyeY -= heightStep;
                orthoEyeY = Math.max(bbMax[1] + 0.5, orthoEyeY);
                console.log(`Ortho eye Y: ${orthoEyeY.toFixed(2)}`);
                break;
        }
        return; // don't fall through to perspective controls
    }
    const moveSpeed = 1.0;
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
        case "w":
            cameraPosition[1] += moveSpeed;
            cameraTarget[1] += moveSpeed;
            console.log("Camera Y:", cameraPosition[1].toFixed(3));
            break;
        case "s":
            cameraPosition[1] -= moveSpeed;
            cameraTarget[1] -= moveSpeed;
            break;
    }
    SORT = true;
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
    // Reconstruction (depth binning + SOR solver) is only meaningful for POINTS mode.
    // Quad modes (DISKS, BILLBOARDS, GAUSSIANS) always do a direct capture.
    const effectiveReconstruction = useReconstruction && currentMode === "POINTS";
    const useParallelBatching = effectiveReconstruction; 
    const activeWorkerCount = useParallelBatching ? CONFIG.workerCount : 1;
    const activeBatchSize = useParallelBatching ? CONFIG.batchSize : Number.MAX_SAFE_INTEGER;
    const colorOrder = effectiveReconstruction || currentMode === "BILLBOARDS" || currentMode === "GAUSSIANS"
        ? CONFIG.reconstructionColorOrder
        : CONFIG.reconstructionColorOrder;

    console.log(
        `🚀 Starting generation (${useParallelBatching ? "parallel + batched" : "single worker, no batching"})...`
    );
    console.log(`Capture mode: ${effectiveReconstruction ? "reconstruction" : "point cloud only"} (mode=${currentMode})`);
    console.log(`Output color order: ${colorOrder.toUpperCase()}`);

    try {
        // Najprej generiramo vse pozicije kamer in naloge, preden začnemo z delom workerjev
        const allTasks = [];
        let imageIndex = 0;

        for (const radius of CONFIG.hemisphereRadii) {
            for (const target of CONFIG.targetPositions) {
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
                        useReconstruction: effectiveReconstruction,
                        colorOrder,
                        radius,
                        mode: currentMode,
                        pointSize: currentPointSize,
                    });
                }

                // Za prvi in drugi radij (zelo oddaljeni) generiramo slike le iz centra
                if (radius === CONFIG.hemisphereRadii[0] || radius === CONFIG.hemisphereRadii[1]) {
                    break;
                } else if (target === CONFIG.targetPositions[0]) {
                    continue;
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
                mode: currentMode,
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
const pointByteSize = 32;
const maxNumberOfPointsPerBuffer = 1024 * 1024;
const numberOfAllPoints = positions.length / 3;
const pointclouds = [];

// Ustvarimo adaptivno mrežo, ki razdeli točke v celice z največ maxNumberOfPointsPerBuffer točkami
// To nam omogoča (za razliko od naključnega deljenja), da nato te celice sortiramo
// glde na povprečno globino točk v celici, kar je potrebno za pravilen izris gausovk (back-to-front)
// Točke najprej sortiramo znotraj vsake celice, nato pa sortiramo še celice med seboj
// Celice niso razporejene v regularno mrežo, ampak so prilagojene gostoti točk
// saj točje v oblaku niso enakomerno proazdeljene - 
// v gostejših delih bo več manjših celic, v redkejših pa manj večjih celic
const grid = new AdaptiveGrid(positions, maxNumberOfPointsPerBuffer);
console.log(`Adaptive grid: ${grid.cells.length} cells`);

for (const cell of grid.cells) {
    const count = cell.indices.length;

    const pointData = new ArrayBuffer(maxNumberOfPointsPerBuffer * pointByteSize);
    const pointDataView = new DataView(pointData);

    cell.indices.forEach((j, slot) => {
        const posIndex = j * 3;
        const pointOffset = slot * pointByteSize;
        pointDataView.setFloat32(pointOffset, positions[posIndex], true);
        pointDataView.setFloat32(pointOffset + 4, positions[posIndex + 1], true);
        pointDataView.setFloat32(pointOffset + 8, positions[posIndex + 2], true);
        pointDataView.setUint32( pointOffset + 12, colors[j], true);
        pointDataView.setFloat32(pointOffset + 16, normals[posIndex], true);
        pointDataView.setFloat32(pointOffset + 20, normals[posIndex + 1], true);
        pointDataView.setFloat32(pointOffset + 24, normals[posIndex + 2], true);
        pointDataView.setFloat32(pointOffset + 28, 0.0, true);
    });

    const pointBuffer = device.createBuffer({
        size: maxNumberOfPointsPerBuffer * pointByteSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pointBuffer, 0, pointData);

    // classification color buffer
    const cellClassColors = new Uint32Array(maxNumberOfPointsPerBuffer);
    cell.indices.forEach((j, slot) => { cellClassColors[slot] = classColors[j]; });
    const classColorBuffer = device.createBuffer({
        size: maxNumberOfPointsPerBuffer * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(classColorBuffer, 0, cellClassColors);

    const renderingBindGroup = device.createBindGroup({
        layout: renderingPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: pointBuffer } },
            { binding: 1, resource: { buffer: classColorBuffer } },
            { binding: 2, resource: { buffer: classUniformBuffer } },
        ],
    });

    // Določimo center vsake celice za kasnejše 
    // sortiranje celic glede na globino (povprečje pozicij točk v celici)
    let cx = 0, cy = 0, cz = 0;
    for (const pi of cell.indices) {
        cx += positions[pi * 3];
        cy += positions[pi * 3 + 1];
        cz += positions[pi * 3 + 2];
    }
    cx /= count; cy /= count; cz /= count;

    pointclouds.push({
        pointBuffer,
        classColorBuffer,
        renderingBindGroup,
        numberOfPoints: count,
        center: [cx, cy, cz], 
    });
}

let depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: "depth32float",
});

let lastSize = { width: null, height: null };

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
prewarmMergeBuffers(numberOfAllPoints);

// Vstvari binding groupe za sortiranje - priprava, lokalno sortiranje in globalno sortiranje
function buildSortBindGroups(pointcloud) {
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

// Pripravimo bind groupe za vsak pointcloud batch
for (const pc of pointclouds) {
    buildSortBindGroups(pc);
}

function sortPointCloud(pointcloud, viewMatrix) {
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

// Debug: prikaži samo en batch, da lažje vidimo sortiranje in izris
let DEBUG_BATCH_INDEX = -1; // -1 = pokaži vse, 0..n = pokaži samo ta batch
document.addEventListener("keydown", (e) => {
    if (e.key === "n") DEBUG_BATCH_INDEX = (DEBUG_BATCH_INDEX + 1) % pointclouds.length;
    if (e.key === "b") DEBUG_BATCH_INDEX = -1; // back to all
    console.log(`Showing batch ${DEBUG_BATCH_INDEX} of ${pointclouds.length}`);
});

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
    let viewMatrix, projectionViewMatrix;

    if (orthoMode) {
        const result = camPositionHelper.computeOrthoTopDownMatrix(
            orthoEyeY, orthoPanX, orthoPanZ, orthoZoom, canvas, bbMin, bbMax
        );
        viewMatrix           = result.viewMatrix;
        projectionViewMatrix = result.projectionViewMatrix;
    } else {
        const result = camPositionHelper.computeCameraMatrix(
            cameraPosition, cameraTarget, canvas, bbMin, bbMax
        );
        viewMatrix           = result.viewMatrix;
        projectionViewMatrix = result.projectionViewMatrix;
    }
    device.queue.writeBuffer(mvpBuffer, 0, new Float32Array(projectionViewMatrix));
    device.queue.writeBuffer(viewMatrixBuffer, 0, new Float32Array(viewMatrix));

    // Depth range: pass huge range so nothing is clipped in preview
    const depthRangeBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(depthRangeBuffer, 0, new Float32Array([0, 1e6]));

    // Target position — vec4f padded for alignment
    const targetPosBuf = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const tp = CONFIG.targetPositions[0];
    device.queue.writeBuffer(targetPosBuf, 0, new Float32Array([tp[0], tp[1], tp[2], 0.0]));

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

    if (currentMode === "POINTS") {
        // Original shader path — nespremenjeno
        const depthRangeBindGroup = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
        });
        const targetPosBindGroup = device.createBindGroup({
            layout: renderingPipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: targetPosBuf } }],
        });
        renderPass.setPipeline(renderingPipeline);
        renderPass.setBindGroup(1, matricesBindGroup);
        renderPass.setBindGroup(2, depthRangeBindGroup);
        renderPass.setBindGroup(3, targetPosBindGroup);
        for (const pointcloud of pointclouds) {
            renderPass.setBindGroup(0, pointcloud.renderingBindGroup);
            renderPass.draw(pointcloud.numberOfPoints);
        }
    } else {
        // Če gre za GAUSSIANS, sortiramo pointcloude (znotraj vsake celice) glede na globino
        if (currentMode === "GAUSSIANS" && SORT) {
            for (const pc of pointclouds) {
                sortPointCloud(pc, projectionViewMatrix);
            }
            SORT = false;
        }
        // Quad shader path — group(3) so SceneParams { targetPosition, cameraPos, pointSize }
        writeSceneParams(tp, orthoMode ? [orthoPanX, orthoEyeY, orthoPanZ] : cameraPosition, currentPointSize);
        const pipeline = quadPipelines[currentMode];
        const depthRangeBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(2),
            entries: [{ binding: 0, resource: { buffer: depthRangeBuffer } }],
        });
        const sceneBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(3),
            entries: [{ binding: 0, resource: { buffer: sceneParamsBuffer } }],
        });
        const matricesQuadBindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: mvpBuffer } },
                { binding: 1, resource: { buffer: viewMatrixBuffer } },
            ],
        });

        // Sortiramo batch-e glede na globino centra batch-a (povprečje pozicij točk v batch-u), 
        // da zagotovimo pravilen back-to-front izris za GAUSSIANS
        const batchesWithDepth = pointclouds.map((pc) => ({
            pc,
            depth: getBatchDepth(pc, projectionViewMatrix),
        }));
        batchesWithDepth.sort((a, b) => b.depth - a.depth);

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(1, matricesQuadBindGroup);
        renderPass.setBindGroup(2, depthRangeBindGroup);
        renderPass.setBindGroup(3, sceneBindGroup);

      // Debugging: če je DEBUG_BATCH_INDEX nastavljen na 0..n, prikažemo samo ta batch, sicer prikažemo vse
       const batchesToRender = DEBUG_BATCH_INDEX === -1
            ? batchesWithDepth
            : [batchesWithDepth[DEBUG_BATCH_INDEX]];

        for (const batch of batchesToRender) {
            const pc = batch.pc;
            renderPass.setBindGroup(0, device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: pc.pointBuffer } }],
            }));
            renderPass.draw(pc.numberOfPoints * 6);
        }
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

canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    if (orthoMode) {
        const factor = e.deltaY > 0 ? 1.1 : 0.9;
        orthoZoom = Math.max(0.5, orthoZoom * factor);
    } else {
        distance = Math.max(0.5, distance + e.deltaY * 0.01);
        updateCameraOrbit();
    }
}, { passive: false });