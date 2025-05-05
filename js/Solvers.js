export class Solver {
	constructor(canvas, device) {
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
		this.jacobianPipeline = null;
		this.residualPipeline = null;
		this.updateRedBlackPipeline = null;
		this.maxIterations = 10000;
	}

	async createJacobianPipeline() {
		const jacobiaCode = await fetch("./shaders/jacobia.wgsl").then((response) =>
			response.text()
		);

		const jacobiaModule = this.device.createShaderModule({
			code: jacobiaCode,
		});

		const jacobiaPipeline = this.device.createComputePipeline({
			compute: {
				module: jacobiaModule,
				entryPoint: "main",
			},
			layout: "auto",
		});
		
		return jacobiaPipeline;
	}

	async createUpdateRedBlackPipeline() {
		const updateRedBlackWGSL = await fetch("./shaders/updateRedBlack.wgsl").then(
			(response) => response.text()
		);
		
		const updateRedBlackModule = this.device.createShaderModule({
			code: updateRedBlackWGSL,
		});
		
		const updateRedBlackPipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: updateRedBlackModule,
				entryPoint: "main",
			},
		});

		return updateRedBlackPipeline;
	}

	async createResidualPipeline() {
		const residualCode = await fetch("./shaders/residual.wgsl").then(response => response.text());
		const residualModule = this.device.createShaderModule({ code: residualCode });

		const residualPipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: residualModule,
				entryPoint: "main"
			}
		});

		return residualPipeline;
	}

	async jacobian(captureTexture, reconstructionRead, reconstructionWrite) {
		// Perform Jacobian operation
		this.jacobianPipeline = await this.createJacobianPipeline();
		const jacobiCommandEncoder = this.device.createCommandEncoder();

		for (let i = 0; i < this.maxIterations; i++) {
			const jacobiaBindGroup = this.device.createBindGroup({
				layout: this.jacobianPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: captureTexture.createView() },
					{ binding: 1, resource: reconstructionRead.createView() },
					{ binding: 2, resource: reconstructionWrite.createView() },
				],
			});
			const jacobiaPassEncoder = jacobiCommandEncoder.beginComputePass();
			jacobiaPassEncoder.setPipeline(this.jacobianPipeline);
			jacobiaPassEncoder.setBindGroup(0, jacobiaBindGroup);
			jacobiaPassEncoder.dispatchWorkgroups(
				Math.ceil(this.width / 8),
				Math.ceil(this.height / 8)
			);
			jacobiaPassEncoder.end();

			[reconstructionRead, reconstructionWrite] = [
				reconstructionWrite,
				reconstructionRead,
			];
		}

		const jacobiaOutputBuffer = this.device.createBuffer({
			size: this.width * this.height * 4, // 4 bytes per pixel (RGBA)
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		jacobiCommandEncoder.copyTextureToBuffer(
			{
				texture: reconstructionRead,
				mipLevel: 0,
				origin: { x: 0, y: 0, z: 0 },
			},
			{
				buffer: jacobiaOutputBuffer,
				bytesPerRow: this.width * 4,
				rowsPerImage: this.height,
			},
			[this.width, this.height, 1]
		);

		this.device.queue.submit([jacobiCommandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		// Read back and save the Jacobian results
		await jacobiaOutputBuffer.mapAsync(GPUMapMode.READ);
		const jacobiaArrayBuffer = jacobiaOutputBuffer.getMappedRange();
		const jacobianImage = new Uint8Array(jacobiaArrayBuffer.slice(0));

		console.log("Expected buffer size:", this.width * this.height * 4);
		console.log("Actual buffer size:", jacobiaOutputBuffer.size);
		console.log("ArrayBuffer length:", jacobiaArrayBuffer.byteLength);
		console.log("Is buffer mapped?", jacobiaOutputBuffer.mapState);
		jacobiaOutputBuffer.unmap();

		return jacobianImage;
	}

	async sorWithResidual(captureTexture, reconstructionRead, reconstructionWrite) {
		this.updateRedBlackPipeline = await this.createUpdateRedBlackPipeline();
		this.residualPipeline = await this.createResidualPipeline();
		const tolerance = 1e-5;
        const omega = 1.5;
        // Create textures for residuals
        const residualTexture = this.device.createTexture({
            size: [this.width, this.height],
            format: 'rgba32float',
            usage: GPUTextureUsage.STORAGE_BINDING | 
                   GPUTextureUsage.TEXTURE_BINDING |
                   GPUTextureUsage.COPY_SRC
        });

        const residualReductionBuffer = this.device.createBuffer({
            size: 4, // For reduction steps
            usage: GPUBufferUsage.STORAGE | 
                   GPUBufferUsage.COPY_SRC |
                   GPUBufferUsage.COPY_DST
        });
		this.device.queue.writeBuffer(residualReductionBuffer, 0, new Uint32Array([0]));

        // Create buffer for red-black control
        const redBlackBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Create buffer for omega value
        const omegaBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(omegaBuffer, 0, new Float32Array([omega]));

        let iteration = 0;
        let residualNorm = Infinity;
        
        while (iteration < this.maxIterations && residualNorm > tolerance) {
            // Alternate between red and black phases
            for (let color = 0; color < 2; color++) {
                this.device.queue.writeBuffer(
                    redBlackBuffer,
                    0,
                    new Uint32Array([color % 2])
                );

                const commandEncoder = this.device.createCommandEncoder();
                
                // SOR update pass
                const updateBindGroup = this.device.createBindGroup({
                    layout: this.updateRedBlackPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: captureTexture.createView() },
                        { binding: 1, resource: reconstructionRead.createView() },
                        { binding: 2, resource: reconstructionWrite.createView() },
                        { binding: 3, resource: { buffer: redBlackBuffer } },
                        { binding: 4, resource: { buffer: omegaBuffer } }
                    ],
                });

                const updatePass = commandEncoder.beginComputePass();
                updatePass.setPipeline(this.updateRedBlackPipeline);
                updatePass.setBindGroup(0, updateBindGroup);
                updatePass.dispatchWorkgroups(
                    Math.ceil(this.width / 8),
                    Math.ceil(this.height / 8)
                );
                updatePass.end();

                // Residual computation pass
                const residualBindGroup = this.device.createBindGroup({
                    layout: this.residualPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: captureTexture.createView() },
                        { binding: 1, resource: reconstructionWrite.createView() },
                        { binding: 2, resource: residualTexture.createView() },
                        { binding: 3, resource: { buffer: residualReductionBuffer } }
                    ],
                });

                const residualPass = commandEncoder.beginComputePass();
                residualPass.setPipeline(this.residualPipeline);
                residualPass.setBindGroup(0, residualBindGroup);
                residualPass.dispatchWorkgroups(
                    Math.ceil(this.width / 8),
                    Math.ceil(this.height / 8)
                );
                residualPass.end();

                this.device.queue.submit([commandEncoder.finish()]);

                // Swap textures for next iteration
                [reconstructionRead, reconstructionWrite] = [
                    reconstructionWrite,
                    reconstructionRead,
                ];
            }

            residualNorm = await this.readResidualNorm(residualReductionBuffer);
            iteration++;
            
            if (iteration % 100 === 0) {
                console.log(`Iteration ${iteration}, residual: ${residualNorm}`);
            }
        }

        console.log(`Converged in ${iteration} iterations with residual ${residualNorm}`);
        
        // Return final result
        const outputBuffer = this.device.createBuffer({
            size: this.width * this.height * 4,
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
                bytesPerRow: this.width * 4,
                rowsPerImage: this.height,
            },
            [this.width, this.height, 1]
        );

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        await outputBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = outputBuffer.getMappedRange();
        const resultImage = new Uint8Array(arrayBuffer.slice(0));
        outputBuffer.unmap();

        residualTexture.destroy();
        residualReductionBuffer.destroy();
        redBlackBuffer.destroy();
        omegaBuffer.destroy();

        return resultImage;
    }

    async sorRedBlack(captureTexture, reconstructionRead, reconstructionWrite) {
        this.updateRedBlackPipeline = await this.createUpdateRedBlackPipeline();
        const omega = 1.9;
    
        // Create buffer for red-black control
        const redBlackBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    
        // Create buffer for omega value
        const omegaBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(omegaBuffer, 0, new Float32Array([omega]));
    
        for (let iteration = 0; iteration < this.maxIterations; iteration++) {
            // Alternate between red and black phases
            for (let color = 0; color < 2; color++) {
                this.device.queue.writeBuffer(
                    redBlackBuffer,
                    0,
                    new Uint32Array([color % 2])
                );
    
                const commandEncoder = this.device.createCommandEncoder();
                
                // SOR update pass
                const updateBindGroup = this.device.createBindGroup({
                    layout: this.updateRedBlackPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: captureTexture.createView() },
                        { binding: 1, resource: reconstructionRead.createView() },
                        { binding: 2, resource: reconstructionWrite.createView() },
                        { binding: 3, resource: { buffer: redBlackBuffer } },
                        { binding: 4, resource: { buffer: omegaBuffer } }
                    ],
                });
    
                const updatePass = commandEncoder.beginComputePass();
                updatePass.setPipeline(this.updateRedBlackPipeline);
                updatePass.setBindGroup(0, updateBindGroup);
                updatePass.dispatchWorkgroups(
                    Math.ceil(this.width / 8),
                    Math.ceil(this.height / 8)
                );
                updatePass.end();
    
                this.device.queue.submit([commandEncoder.finish()]);
    
                // Swap textures for next iteration
                [reconstructionRead, reconstructionWrite] = [
                    reconstructionWrite,
                    reconstructionRead,
                ];
            }
    
            // Optional: Add progress logging
            if (iteration % 100 === 0) {
                console.log(`Completed ${iteration} iterations`);
            }
        }
    
        // Return final result
        const outputBuffer = this.device.createBuffer({
            size: this.width * this.height * 4,
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
                bytesPerRow: this.width * 4,
                rowsPerImage: this.height,
            },
            [this.width, this.height, 1]
        );
    
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    
        await outputBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = outputBuffer.getMappedRange();
        const resultImage = new Uint8Array(arrayBuffer.slice(0));
        outputBuffer.unmap();
    
        return resultImage;
    }

	async readResidualNorm(buffer) {
        // In a real implementation, you'd use parallel reduction
        // For simplicity, we'll just read a single value here
        const readbackBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(buffer, 0, readbackBuffer, 0, 4);
        this.device.queue.submit([commandEncoder.finish()]);

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const residual = new Uint32Array(readbackBuffer.getMappedRange())[0];
        readbackBuffer.unmap();
        readbackBuffer.destroy();

        const residualNorm = residual / 1000.0;
    	return Math.sqrt(residualNorm / (this.width * this.height));
    }
}
