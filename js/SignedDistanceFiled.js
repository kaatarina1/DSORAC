import { PriorityQueue } from "./PriorityQueue.js";
import { saveSDFToPNG, saveTextureToPNG } from "./Utils.js";

export class SignedDistanceFiled {
    constructor(device, texture, width, height) {
        this.device = device;
        this.texture = texture;
        this.width = width;
        this.height = height;
        this.closed = new Uint32Array(width * height);
        this.distance = new Float32Array(width * height);
        this.distance.fill(width * 2);
        this.priorityQueue = new PriorityQueue();
        this.closedPipeline = null;
    }

    async createClosedPipeline() {
        const closedCode = await fetch("shaders/isColored.wgsl").then(
			(res) => res.text()
		);
		const closedModule = this.device.createShaderModule({
			code: closedCode,
		});
		return this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: closedModule,
				entryPoint: "main",
			},
		});
    }


    async seed() {
        this.closedPipeline = await this.createClosedPipeline();
        const closedTexture = this.device.createTexture({
            size: [this.width, this.height],
            format: "r32uint",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });

        const closedBuffer = this.device.createBuffer({
            size: this.closed.byteLength,
            usage: GPUBufferUsage.COPY_DST |GPUBufferUsage.MAP_READ,
        });

        const closedBinding = this.device.createBindGroup({
            layout: this.closedPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.texture.createView(),
                },
                {
                    binding: 1,
                    resource: closedTexture.createView(),
                },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();  
        passEncoder.setPipeline(this.closedPipeline);
        passEncoder.setBindGroup(0, closedBinding);
        passEncoder.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        passEncoder.end();

        commandEncoder.copyTextureToBuffer(
            {
                texture: closedTexture,
                mipLevel: 0,
                origin: { x: 0, y: 0, z: 0 },
            },
            {
                buffer: closedBuffer,
                offset: 0,
                bytesPerRow: this.width * 4,
                rowsPerImage: this.height,  
            },
            {
                width: this.width,
                height: this.height,
                depthOrArrayLayers: 1,
            }
        );
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

		await closedBuffer.mapAsync(GPUMapMode.READ);
		const outputArrayBuffer = closedBuffer.getMappedRange();
		this.closed = new Uint32Array(outputArrayBuffer.slice(0));

        for (let i = 0; i < this.closed.length; i++) {
            if (this.closed[i] != 0) {
                this.distance[i] = 0;
                this.addNeighbors(i, 0, 0);
            }
        }
    }

    propagation() {
        let maxDistance = 0;
        while (!this.priorityQueue.isEmpty()) {
            let element = this.priorityQueue.dequeue();
            let index = element.index;
            if (this.closed[index] == 1) {
                continue;
            }
            this.closed[index] = 1;
            this.distance[index] = element.distance;
            if (element.distance > maxDistance) {
                maxDistance = element.distance;
            }
            this.addNeighbors(index, element.dx, element.dy);
        }
        const sdfTextureData = new Float32Array(this.distance.length * 4);
        let index = 0;
        for (let i = 0; i < this.distance.length; i++) {
            this.distance[i] = this.distance[i] / maxDistance;
            sdfTextureData[index] = 1.0;
            sdfTextureData[index + 1] = 1.0;
            sdfTextureData[index + 2] = 1.0;
            sdfTextureData[index + 3] = this.distance[i];
            index += 4;
        }

        const sdfTexture = this.device.createTexture({
            size: [this.width, this.height],
            format: "rgba32float",
            usage: GPUTextureUsage.STORAGE_BINDING |
			       GPUTextureUsage.TEXTURE_BINDING |
			       GPUTextureUsage.COPY_DST |
			       GPUTextureUsage.COPY_SRC,
        });

        const sdfBuffer = this.device.createBuffer({
            size: sdfTextureData.byteLength,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE,
        });

        this.device.queue.writeTexture(
            { texture: sdfTexture },
            sdfTextureData,
            { bytesPerRow: this.width * 4 * 4 },
            { width: this.width, height: this.height }
        );

        return sdfTexture;
    }

    euclideanDistance(dx, dy) {
        return Math.sqrt(dx * dx + dy * dy);
    }

    addNeighbors(i, dx, dy) {
        let firstNeighbor = i - 1;
        let secondNeigbbor = firstNeighbor - this.width;
        let thirdNeighbor = secondNeigbbor + 1;
        let fourthNeighbor = thirdNeighbor + 1;
        let fifthNeighbor = i + 1;
        let sixthNeighbor = fifthNeighbor + this.width;
        let seventhNeighbor = sixthNeighbor - 1;
        let eighthNeighbor = seventhNeighbor - 1;
        if (i == 0) {
            // has only 5th, 6th, 7th neighbors
            if (this.closed[fifthNeighbor] == 0) {
                this.priorityQueue.enqueue(fifthNeighbor, dx + 1, dy, this.euclideanDistance(dx + 1, dy));
            } 
            if (this.closed[sixthNeighbor] == 0) {
                this.priorityQueue.enqueue(sixthNeighbor, dx + 1, dy + 1, this.euclideanDistance(dx + 1, dy + 1));
            }
            if (this.closed[seventhNeighbor] == 0) {
                this.priorityQueue.enqueue(seventhNeighbor, dx, dy + 1, this.euclideanDistance(dx, dy + 1));
            }
        } else if (i == this.width - 1) {
            // has only 1st, 7th, 8th neighbors
            if (this.closed[firstNeighbor] == 0) {
                this.priorityQueue.enqueue(firstNeighbor, dx - 1, dy, this.euclideanDistance(dx - 1, dy));
            }
            if (this.closed[eighthNeighbor] == 0) {
                this.priorityQueue.enqueue(eighthNeighbor, dx - 1, dy + 1, this.euclideanDistance(dx - 1, dy + 1));
            }
            if (this.closed[seventhNeighbor] == 0) {
                this.priorityQueue.enqueue(seventhNeighbor, dx, dy + 1, this.euclideanDistance(dx, dy + 1));
            }
        } else if (i == this.closed.length - this.width) {
            // has only 3rd, 4th, 5th neighbors
            if (this.closed[thirdNeighbor] == 0) {
                this.priorityQueue.enqueue(thirdNeighbor, dx, dy - 1, this.euclideanDistance(dx, dy - 1));
            }
            if (this.closed[fourthNeighbor] == 0) {
                this.priorityQueue.enqueue(fourthNeighbor, dx + 1, dy - 1, this.euclideanDistance(dx + 1, dy - 1));
            }
            if (this.closed[fifthNeighbor] == 0) {
                this.priorityQueue.enqueue(fifthNeighbor, dx + 1, dy, this.euclideanDistance(dx + 1, dy));
            } 
        } else if (i == this.closed.length - 1) {
            // has only 1st, 2nd, 3rd neighbors
            if (this.closed[firstNeighbor] == 0) {
                this.priorityQueue.enqueue(firstNeighbor, dx - 1, dy, this.euclideanDistance(dx - 1, dy));
            }
            if (this.closed[secondNeigbbor] == 0) {
                this.priorityQueue.enqueue(secondNeigbbor, dx - 1, dy - 1, this.euclideanDistance(dx - 1, dy - 1));
            }
            if (this.closed[thirdNeighbor] == 0) {
                this.priorityQueue.enqueue(thirdNeighbor, dx, dy - 1, this.euclideanDistance(dx, dy - 1));
            }
        } else if (i % this.width == 0) {
            // has 3rd, 4th, 5th, 6th and 7th neighbors
            if (this.closed[thirdNeighbor] == 0) {
                this.priorityQueue.enqueue(thirdNeighbor, dx, dy - 1, this.euclideanDistance(dx, dy - 1));
            }
            if (this.closed[fourthNeighbor] == 0) {
                this.priorityQueue.enqueue(fourthNeighbor, dx + 1, dy - 1, this.euclideanDistance(dx + 1, dy - 1));
            }
            if (this.closed[fifthNeighbor] == 0) {
                this.priorityQueue.enqueue(fifthNeighbor, dx + 1, dy, this.euclideanDistance(dx + 1, dy));
            } 
            if (this.closed[sixthNeighbor] == 0) {
                this.priorityQueue.enqueue(sixthNeighbor, dx + 1, dy + 1, this.euclideanDistance(dx + 1, dy + 1));
            }
            if (this.closed[seventhNeighbor] == 0) {
                this.priorityQueue.enqueue(seventhNeighbor, dx, dy + 1, this.euclideanDistance(dx, dy + 1));
            }
        } else if (i + 1 % this.width == 0) {
            // has 1st, 2nd, 3rd, 7th and 8th neighbors
            if (this.closed[firstNeighbor] == 0) {
                this.priorityQueue.enqueue(firstNeighbor, dx - 1, dy, this.euclideanDistance(dx - 1, dy));
            }
            if (this.closed[secondNeigbbor] == 0) {
                this.priorityQueue.enqueue(secondNeigbbor, dx - 1, dy - 1, this.euclideanDistance(dx - 1, dy - 1));
            }
            if (this.closed[thirdNeighbor] == 0) {
                this.priorityQueue.enqueue(thirdNeighbor, dx, dy - 1, this.euclideanDistance(dx, dy - 1));
            }
            if (this.closed[seventhNeighbor] == 0) {
                this.priorityQueue.enqueue(seventhNeighbor, dx, dy + 1, this.euclideanDistance(dx, dy + 1));
            }
            if (this.closed[eighthNeighbor] == 0) {
                this.priorityQueue.enqueue(eighthNeighbor, dx - 1, dy + 1, this.euclideanDistance(dx - 1, dy + 1));
            }
        } else {
            // has all 8 neighbors
            if (this.closed[firstNeighbor] == 0) {
                this.priorityQueue.enqueue(firstNeighbor, dx - 1, dy, this.euclideanDistance(dx - 1, dy));
            }
            if (this.closed[secondNeigbbor] == 0) {
                this.priorityQueue.enqueue(secondNeigbbor, dx - 1, dy - 1, this.euclideanDistance(dx - 1, dy - 1));
            }
            if (this.closed[thirdNeighbor] == 0) {
                this.priorityQueue.enqueue(thirdNeighbor, dx, dy - 1, this.euclideanDistance(dx, dy - 1));
            }
            if (this.closed[fourthNeighbor] == 0) {
                this.priorityQueue.enqueue(fourthNeighbor, dx + 1, dy - 1, this.euclideanDistance(dx + 1, dy - 1));
            }
            if (this.closed[fifthNeighbor] == 0) {
                this.priorityQueue.enqueue(fifthNeighbor, dx + 1, dy, this.euclideanDistance(dx + 1, dy));
            } 
            if (this.closed[sixthNeighbor] == 0) {
                this.priorityQueue.enqueue(sixthNeighbor, dx + 1, dy + 1, this.euclideanDistance(dx + 1, dy + 1));
            }
            if (this.closed[seventhNeighbor] == 0) {
                this.priorityQueue.enqueue(seventhNeighbor, dx, dy + 1, this.euclideanDistance(dx, dy + 1));
            }
            if (this.closed[eighthNeighbor] == 0) {
                this.priorityQueue.enqueue(eighthNeighbor, dx - 1, dy + 1, this.euclideanDistance(dx - 1, dy + 1));
            }
        }
    }

    async generateSDF() {
        await this.seed();
        return this.propagation();
    }
}