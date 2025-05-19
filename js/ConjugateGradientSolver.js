import { Solver } from "./Solvers";
import { convertTexture } from "./Utils";

export class ConjugateGradientSolver {
    constructor(canvas, device, iterations = 1000) {
        this.canvas = canvas;
        this.device = device;
        this.iterations = iterations;
        this.format = "rgba32float"; 
        this.debug = true; // Enable debugging for troubleshooting
        
        this.width = canvas.width;
        this.height = canvas.height;
        this.laplacianPipeline = null;
        this.updatePipeline = null;
        this.residualPipeline = null;
        this.dotProductPipeline = null;

        this.residualCurrent = null;
        this.residualNext = null;
        this.conjGradientCurrent = null;
        this.conjGradientNext = null;
        this.cgTempDrawing = null;
    }

    async createLaplacianPipeline() {
        const laplacianCode = await fetch("shaders/laplacian.wgsl").then((res) => res.text());
        const laplacianModule = this.device.createShaderModule({
            code: laplacianCode,
        });
        return this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: laplacianModule,
                entryPoint: "main",
            },
        });
    }

    async createUpdatePipeline() {
        const updateCode = await fetch("shaders/update.wgsl").then((res) => res.text());
        const updateModule = this.device.createShaderModule({
            code: updateCode,
        });
        return this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: updateModule,
                entryPoint: "main",
            },
        });
    }

    async createResidualPipeline() {
        const residualCode = await fetch("shaders/residual.wgsl").then((res) => res.text());
        const residualModule = this.device.createShaderModule({
            code: residualCode,
        });
        return this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: residualModule,
                entryPoint: "main",
            },
        });
    }

    async createDotProductPipeline() {
        const dotProductCode = await fetch("shaders/dotProduct.wgsl").then((res) => res.text());
        const dotProductModule = this.device.createShaderModule({
            code: dotProductCode,
        });
        return this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: dotProductModule,
                entryPoint: "main",
            },
        });
    }

    async initialize() {
        this.laplacianPipeline = await this.createLaplacianPipeline();
        this.updatePipeline = await this.createUpdatePipeline();
        this.residualPipeline = await this.createResidualPipeline();
        this.dotProductPipeline = await this.createDotProductPipeline();

        this.residualCurrent = this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
        });

        this.residualNext = this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
        });

        this.conjGradientCurrent = this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
        });

        this.conjGradientNext = this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
        });

        this.cgTempDrawing = this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
			GPUTextureUsage.COPY_DST |
			GPUTextureUsage.COPY_SRC,
        });
    }

    async conjGradientSolve(points, reconstructionWrite) {
        await this.initialize();

        await this.laplatian(this.conjGradientCurrent, points, this.cgTempDrawing);
        await this.computeResidual(this.cgTempDrawing, points, this.residualCurrent);
        await this.copyTexture(this.residualCurrent, this.conjGradientCurrent);

        // Debug: Check if residual has values
        if (this.debug) {
            console.log("Initial residual values:");
            await this.checkTextureValues(this.residualCurrent);

            // Debug: Check if search direction has values
            console.log("Initial search direction values:");
            await this.checkTextureValues(this.conjGradientCurrent);
        }

        let reconstructionRead = this.device.createTexture({
            size: [this.width, this.height],
            format: 'rgba32float',
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
        });
            
        for (let i = 0; i < this.iterations; i++) {
            await this.laplatian(this.conjGradientCurrent, points, this.cgTempDrawing);
            
            const rr = await this.dotProduct(this.residualCurrent, this.residualCurrent);
            const pAp = await this.dotProduct(this.conjGradientCurrent, this.cgTempDrawing);
            
            if (this.debug && i % 100 === 0) {
                console.log(`Iteration ${i} - rrDot:`, rr);
                console.log(`Iteration ${i} - pAp:`, pAp);
            }

            if ((pAp.reduce((acc, n) => acc * n, 1) || 0) === 0) {
                continue;
            }
            
            if (this.debug && i % 100 === 0) {
                console.log(`Iteration ${i} - alpha:`, alpha);
            }

            this.alpha = rr.map((n, i) => n / pAp[i]);
            await this.update(reconstructionRead, this.conjGradientCurrent, reconstructionWrite, alpha);
            
            const negAlpha = alpha.map(a => -a);
            await this.update(this.residualCurrent, this.cgTempDrawing, this.residualNext, negAlpha);
            
            const rrNew = await this.dotProduct(this.residualNext, this.residualNext);
            let beta = rrNew.map((n, i) => n / rr[i]);
            
            if (this.debug && i % 100 === 0) {
                console.log(`Iteration ${i} - rrNew:`, rrNew);
                console.log(`Iteration ${i} - beta:`, beta);
            }
            
            // Update p: p_new = r_new + beta * p_old
            await this.update(this.residualNext, this.conjGradientCurrent, this.conjGradientNext, beta);
            
            // Swap buffers for next iteration
            [this.residualCurrent, this.residualNext] = [this.residualNext, this.residualCurrent];
            [this.conjGradientCurrent, this.conjGradientNext] = [this.conjGradientNext, this.conjGradientCurrent];
            [reconstructionRead, reconstructionWrite] = [reconstructionWrite, reconstructionRead];
            
            // Output progress
            if (i % 10 === 0) {
                console.log(`CG iteration ${i}/${this.iterations}`);
            }
        }

        // Copy the final solution to an output buffer
        const outputBuffer = this.device.createBuffer({
            size: this.width * this.height * 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            {
                texture: reconstructionRead,
                mipLevel: 0,
                origin: { x: 0, y: 0, z: 0 },
            },
            {
                buffer: outputBuffer,
                offset: 0,
                bytesPerRow: 16 * this.width,
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

		await outputBuffer.mapAsync(GPUMapMode.READ);
		const outputArrayBuffer = outputBuffer.getMappedRange();
		const floatData = new Float32Array(outputArrayBuffer.slice(0));

        // Convert to Uint8Array for display
        const outputImage = new Uint8Array(this.width * this.height * 4);
        for (let i = 0; i < floatData.length; i += 4) {
            // Simple normalization for display
            outputImage[i] = Math.max(0, Math.min(255, floatData[i+2] * 255));
            outputImage[i+1] = Math.max(0, Math.min(255, floatData[i+1] * 255));
            outputImage[i+2] = Math.max(0, Math.min(255, floatData[i] * 255)); 
            outputImage[i+3] = Math.max(0, Math.min(255, floatData[i+3] * 255));
        }

        outputBuffer.unmap();
        return outputImage;
    }

    async dotProduct(textureA, textureB) {
        const outputBuffer = this.device.createBuffer({
            size: this.width * this.height * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const bindGroup = this.device.createBindGroup({
            layout: this.dotProductPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: textureA.createView() },
                { binding: 1, resource: textureB.createView() },
                { binding: 2, resource: { buffer: outputBuffer } },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.dotProductPipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        passEncoder.end();

        // Read the output buffer
        const readBuffer = this.device.createBuffer({
            size: this.width * this.height * 16,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, this.width * this.height * 16);
        this.device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readBuffer.getMappedRange();
        const partialResults = new Float32Array(arrayBuffer);

        // Aggregate results
        const finalResults = [0, 0, 0, 0]; // [R, G, B, A]
        for (let i = 0; i < partialResults.length; i += 4) {
            finalResults[0] += partialResults[i + 0]; // Sum R
            finalResults[1] += partialResults[i + 1]; // Sum G
            finalResults[2] += partialResults[i + 2]; // Sum B
            finalResults[3] += partialResults[i + 3]; // Sum A
        }

        return finalResults;
    }

    async computeResidual(reconstructionRead, points, residualOutput) {
        const residualReductionBuffer = this.device.createBuffer({
            size: 4, // For reduction steps
            usage: GPUBufferUsage.STORAGE | 
                   GPUBufferUsage.COPY_SRC |
                   GPUBufferUsage.COPY_DST
        });
		this.device.queue.writeBuffer(residualReductionBuffer, 0, new Uint32Array([0]));

        const commandEncoder = this.device.createCommandEncoder();
        const bindingGroup = this.device.createBindGroup({
            layout: this.residualPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: points.createView(),
                },
                {
                    binding: 1,
                    resource: reconstructionRead.createView(),
                },
                {
                    binding: 2,
                    resource: residualOutput.createView(),
                },
                {
                    binding: 3,
                    resource: { buffer: residualReductionBuffer },
                },
            ],
        });
        
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.residualPipeline);
        pass.setBindGroup(0, bindingGroup);
        pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        pass.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    async laplatian(inputTex, pointsTex, outputTex) {
        const commandEncoder = this.device.createCommandEncoder();
        
        // Get the binding layout directly from the pipeline
        const layout = this.laplacianPipeline.getBindGroupLayout(0);
        
        const bindingGroup = this.device.createBindGroup({
            layout: layout,
            entries: [
                {
                    binding: 0,
                    resource: pointsTex.createView(),
                },
                {
                    binding: 1,
                    resource: inputTex.createView(),
                },
                {
                    binding: 2,
                    resource: outputTex.createView(),
                },
            ],
        });

        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.laplacianPipeline);
        pass.setBindGroup(0, bindingGroup);
        pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        pass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    async update(lhs, rhs, write, alpha) {
        const alphaBuffer = this.device.createBuffer({
            size: 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        
        this.device.queue.writeBuffer(
            alphaBuffer,
            0,
            new Float32Array(alpha)
        );

        const commandEncoder = this.device.createCommandEncoder();
        
        // Get the binding layout directly from the pipeline
        const layout = this.updatePipeline.getBindGroupLayout(0);
        
        const bindingGroup = this.device.createBindGroup({
            layout: layout,
            entries: [
                {
                    binding: 0,
                    resource: lhs.createView(),
                },
                {
                    binding: 1,
                    resource: rhs.createView(),
                },
                {
                    binding: 2,
                    resource: write.createView(),
                },
                {
                    binding: 3,
                    resource: { buffer: alphaBuffer },
                },
            ],
        });

        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.updatePipeline);
        pass.setBindGroup(0, bindingGroup);
        pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        pass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    async copyTexture(fromTexture, toTexture) {
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyTextureToTexture(
            { texture: fromTexture },
            { texture: toTexture },
            [fromTexture.width || this.width, fromTexture.height || this.height, 1]
        );
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }

    // Used for debugging
    async checkTextureValues(texture) {
        const stagingBuffer = this.device.createBuffer({
            size: this.width * this.height * 4 * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            {
                texture: texture,
                mipLevel: 0,
                origin: { x: 0, y: 0, z: 0 },
            },
            {
                buffer: stagingBuffer,
                bytesPerRow: this.width * 16,
                rowsPerImage: this.height,
            },
            [this.width, this.height, 1]
        );
        this.device.queue.submit([commandEncoder.finish()]);

        // Ensure the buffer is mapped for reading
        await stagingBuffer.mapAsync(GPUMapMode.READ);

        // Access the buffer data
        const arrayBuffer = stagingBuffer.getMappedRange();

        // Interpret the buffer data
        const pointData = new DataView(arrayBuffer);

        let nonZeroCount = 0;
        let totalPixels = this.width * this.height;
        let maxValue = 0;

        for (let i = 0; i < totalPixels; i++) {
            const baseOffset = i * 16;
            const r = pointData.getFloat32(baseOffset, true); 
            const g = pointData.getFloat32(baseOffset + 4, true); 
            const b = pointData.getFloat32(baseOffset + 8, true); 
            const a = pointData.getFloat32(baseOffset + 12, true); 

            const magnitude = Math.sqrt(r*r + g*g + b*b + a*a);
            maxValue = Math.max(maxValue, magnitude);

            if (r !== 0 || g !== 0 || b !== 0 || a !== 0) {
                nonZeroCount++;
                if (nonZeroCount <= 5) {
                    console.log(`Pixel ${i}: (${r}, ${g}, ${b}, ${a})`);
                }
            }
        }
        
        console.log(`Non-zero pixels: ${nonZeroCount} out of ${totalPixels} (${(nonZeroCount/totalPixels*100).toFixed(2)}%)`);
        console.log(`Maximum value: ${maxValue}`);
        
        // Unmap the buffer
        stagingBuffer.unmap();
    }
}