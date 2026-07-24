// js/WorkerPool.js - Manages multiple workers for parallel image generation

export class WorkerPool {
    constructor(workerCount = navigator.hardwareConcurrency || 4) {
        this.workerCount = Math.min(workerCount, navigator.hardwareConcurrency || 4);
        this.workers = [];
        this.taskQueue = [];
        this.activeWorkers = new Set();
        this.results = new Map();
        this.initialized = false;
    }

    async initialize(config) {
        console.log(`Initializing ${this.workerCount} workers...`);
        
        const initPromises = [];
        
        for (let i = 0; i < this.workerCount; i++) {
            const worker = new Worker('./imageGenerationWorker.js', { type: 'module' });
            
            const initPromise = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error(`Worker ${i} initialization timeout`));
                }, 30000);

                worker.onmessage = (e) => {
                    if (e.data.type === 'INIT_COMPLETE') {
                        clearTimeout(timeout);
                        resolve();
                    } else if (e.data.type === 'ERROR') {
                        clearTimeout(timeout);
                        reject(new Error(e.data.error));
                    }
                };

                worker.onerror = (error) => {
                    clearTimeout(timeout);
                    reject(error);
                };
            });

            worker.postMessage({ type: 'INIT', data: config });
            this.workers.push({
                worker,
                id: i,
                busy: false,
                initPromise
            });

            initPromises.push(initPromise);
        }

        try {
            await Promise.all(initPromises);
            this.initialized = true;
            console.log(`All ${this.workerCount} workers initialized successfully`);
        } catch (error) {
            console.error('Worker initialization failed:', error);
            throw error;
        }
    }

    async processImages(tasks, progressCallback) {
        if (!this.initialized) {
            throw new Error('WorkerPool not initialized. Call initialize() first.');
        }

        return new Promise((resolve, reject) => {
            const results = [];
            let completed = 0;
            let failed = 0;
            const totalTasks = tasks.length;

            // Set up message handlers for all workers
            this.workers.forEach(({ worker, id }) => {
                worker.onmessage = (e) => {
                    const { type, data } = e.data;

                    switch (type) {
                        case 'IMAGE_COMPLETE':
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

                            // Process next task
                            this.assignNextTask(id);
                            break;

                        case 'PROGRESS':
                            if (progressCallback) {
                                progressCallback({
                                    type: 'layer_progress',
                                    imageIndex: e.data.imageIndex,
                                    layer: e.data.layer,
                                    totalLayers: e.data.totalLayers,
                                    workerId: id
                                });
                            }
                            break;

                        case 'SAVE_PNG': {
                            // Worker ne more sprožiti prenosa — to naredimo tukaj
                            const url = URL.createObjectURL(e.data.blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = e.data.fileName;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                            break;
                        }

                        case 'ERROR':
                            failed++;
                            console.error(`Worker ${id} error on image ${e.data.imageIndex}:`, e.data.error);
                            
                            // Still try to continue with other tasks
                            this.assignNextTask(id);
                            break;
                    }

                    // Check if all tasks are complete
                    if (completed + failed >= totalTasks) {
                        if (failed > 0) {
                            console.warn(`Completed with ${failed} failures out of ${totalTasks} tasks`);
                        }
                        resolve(results);
                    }
                };

                worker.onerror = (error) => {
                    console.error(`Worker ${id} fatal error:`, error);
                    failed++;
                    
                    if (completed + failed >= totalTasks) {
                        reject(error);
                    }
                };
            });

            // Add all tasks to queue
            this.taskQueue = [...tasks];

            // Start initial batch of tasks
            this.workers.forEach(({ id }) => {
                this.assignNextTask(id);
            });
        });
    }

    assignNextTask(workerId) {
        const workerInfo = this.workers[workerId];
        
        if (this.taskQueue.length === 0) {
            workerInfo.busy = false;
            return;
        }

        const task = this.taskQueue.shift();
        workerInfo.busy = true;
        workerInfo.worker.postMessage({
            type: 'GENERATE_IMAGE',
            data: task
        });
    }

    async shutdown() {
        console.log('Shutting down worker pool...');
        
        for (const { worker } of this.workers) {
            worker.postMessage({ type: 'SHUTDOWN' });
            worker.terminate();
        }
        
        this.workers = [];
        this.taskQueue = [];
        this.initialized = false;
        
        console.log('Worker pool shut down');
    }

    getStats() {
        return {
            totalWorkers: this.workerCount,
            busyWorkers: this.workers.filter(w => w.busy).length,
            queueLength: this.taskQueue.length,
            initialized: this.initialized
        };
    }
}
