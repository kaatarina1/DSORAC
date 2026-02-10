import { saveTextureToPNG } from "./Utils";

export class Composer {
	constructor(document, canvas, device, format) {
        this.document = document;
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
        this.format = format;
		
        this.pipeline = this.createPipeline();

        this.depthPoints = [];
        this.reconstructions = [];
        this.sdfs = [];
		this.depths = [];

        this.backgroundTexture = null;
        this.compositeResult = null;
	}

    async createCompositePipeline() {
		const compositeCode = await fetch("./shaders/composite.wgsl").then(
			(response) => response.text()
		);

		const compositeModule = this.device.createShaderModule({
			code: compositeCode,
		});

		const compositePipeline = this.device.createComputePipeline({
			compute: {
				module: compositeModule,
				entryPoint: "main",
			},
			layout: "auto",
		});

		return compositePipeline;
	}

    createPipeline() {
        const vertexShaderCode = `
                        struct VertexOutput {
                            @builtin(position) position: vec4f,
                            @location(0) texCoord: vec2f,
                        };
                        
                        @vertex
                        fn main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                            var pos = array<vec2f, 6>(
                                vec2f(-1.0, -1.0),
                                vec2f(1.0, -1.0),
                                vec2f(-1.0, 1.0),
                                vec2f(-1.0, 1.0),
                                vec2f(1.0, -1.0),
                                vec2f(1.0, 1.0)
                            );
                            
                            var texCoord = array<vec2f, 6>(
                                vec2f(0.0, 1.0),
                                vec2f(1.0, 1.0),
                                vec2f(0.0, 0.0),
                                vec2f(0.0, 0.0),
                                vec2f(1.0, 1.0),
                                vec2f(1.0, 0.0)
                            );
                            
                            var output: VertexOutput;
                            output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
                            output.texCoord = texCoord[vertexIndex];
                            return output;
                        }
                    `;
        const fragmentShaderCode = `
                        @group(0) @binding(0) var backgroundTex: texture_2d<f32>;
                        @group(0) @binding(1) var foregroundTex: texture_2d<f32>;
                        @group(0) @binding(2) var texSampler: sampler;
                        
                        @fragment
                        fn main(@location(0) texCoord: vec2f) -> @location(0) vec4f {
                            let background = textureSample(backgroundTex, texSampler, texCoord);
                            let foreground = textureSample(foregroundTex, texSampler, texCoord);
                            
                            // Simple alpha blending: dst = src * srcAlpha + dst * (1 - srcAlpha)
                            return vec4f(
                                foreground.rgb * foreground.a + background.rgb * (1.0 - foreground.a),
                                foreground.a + background.a * (1.0 - foreground.a)
                            );
                        }
                    `;

        return this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.device.createShaderModule({
                    code: vertexShaderCode
                }),
                entryPoint: 'main'
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: fragmentShaderCode
                }),
                entryPoint: 'main',
                targets: [{
                    format: this.format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });
    }

    // Function to create an initial composite texture from the background
    initializeCompositeTexture() {
        if (!this.backgroundTexture) {
            // Create a black background if no image is loaded
            this.compositeResult = this.device.createTexture({
                size: [this.width, this.height],
                format: this.format,
                usage: 
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.RENDER_ATTACHMENT
            });
            
            // Clear to black
            const commandEncoder = this.device.createCommandEncoder();
            const renderPass = commandEncoder.beginRenderPass({
                colorAttachments: [{
                    view: this.compositeResult.createView(),
                    loadOp: "clear",
                    clearValue: [0, 0, 0, 1], // Black background
                    storeOp: "store"
                }]
            });
            renderPass.end();
            this.device.queue.submit([commandEncoder.finish()]);
        } else {
            // Copy the background to be the initial composite result
            this.compositeResult = this.device.createTexture({
                size: [this.width, this.height],
                format: this.format,
                usage: 
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.RENDER_ATTACHMENT
            });
            
            const commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyTextureToTexture(
                { texture: this.backgroundTexture },
                { texture: this.compositeResult },
                [this.width, this.height]
            );
            this.device.queue.submit([commandEncoder.finish()]);
        }
        
        return this.compositeResult;
    }

    async compositeLayer(layerData) {
        // Create a temporary texture for the new layer
        const layerTexture = this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: 
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT
        });
        
        // Upload the layer data to the texture
        this.device.queue.writeTexture(
            { texture: layerTexture },
            layerData,
            { bytesPerRow: this.width * 4 },
            [this.width, this.height]
        );
        
        // Create a compute shader for alpha compositing
        // This assumes you have a compute pipeline for compositing or you'll need to create one
        // For simplicity, we'll use a render pass with blending enabled
        
        const tempTexture = this.device.createTexture({
            size: [this.width, this.height],
            format: this.format,
            usage: 
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC |
                GPUTextureUsage.RENDER_ATTACHMENT
        });
        
        // Copy the current composite result to the temp texture
        const copyEncoder = this.device.createCommandEncoder();
        copyEncoder.copyTextureToTexture(
            { texture: this.compositeResult },
            { texture: tempTexture },
            [this.width, this.height]
        );
        this.device.queue.submit([copyEncoder.finish()]);
        
        // Create sampler
        const sampler = this.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear'
        });
        
        // Create bind group for compositing
        const compositeBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: tempTexture.createView()
                },
                {
                    binding: 1,
                    resource: layerTexture.createView()
                },
                {
                    binding: 2,
                    resource: sampler
                }
            ]
        });
        
        // Perform the compositing operation
        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: this.compositeResult.createView(),
                loadOp: 'clear',
                clearValue: [0, 0, 0, 0],
                storeOp: 'store'
            }]
        });
        
        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, compositeBindGroup);
        renderPass.draw(6);  // Draw a quad
        renderPass.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
        
        // Clean up temporary resources
        layerTexture.destroy();
        tempTexture.destroy();
    }

    // Function to load a background image
    async loadBackgroundImage(imageUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                // Create a canvas to draw the image
                const tempCanvas = this.document.createElement('canvas');
                tempCanvas.width = this.width;
                tempCanvas.height = this.height;
                const ctx = tempCanvas.getContext('2d');
                
                // Draw the image, stretching it to fit the canvas dimensions
                ctx.drawImage(img, 0, 0, this.width, this.height);
                
                // Get image data
                const imageData = ctx.getImageData(0, 0, this.width, this.height);
                const rgbaData = imageData.data;
    
                // Convert RGBA to BGRA
                const bgraData = new Uint8Array(rgbaData.length);
                for (let i = 0; i < rgbaData.length; i += 4) {
                    bgraData[i] = rgbaData[i + 2];     // B <- R
                    bgraData[i + 1] = rgbaData[i + 1]; // G <- G
                    bgraData[i + 2] = rgbaData[i];     // R <- B
                    bgraData[i + 3] = rgbaData[i + 3]; // A <- A
                }
                
                // Create a WebGPU texture from the swizzled data
                this.backgroundTexture = this.device.createTexture({
                    size: [this.width, this.height],
                    format: this.format,
                    usage: 
                        GPUTextureUsage.TEXTURE_BINDING |
                        GPUTextureUsage.COPY_DST |
                        GPUTextureUsage.COPY_SRC |
                        GPUTextureUsage.STORAGE_BINDING
                });
    
                // Write the swizzled data to the texture
                this.device.queue.writeTexture(
                    { texture: this.backgroundTexture },
                    bgraData,
                    { bytesPerRow: this.width * 4 },
                    [this.width, this.height]
                );
    
                console.log("Background image loaded and swizzled to BGRA");
                resolve();
            };
            
            img.onerror = (event) => {
                console.error(`Error loading image from URL: ${imageUrl}`);
                console.error("Image Error Event:", event);
                console.error("Browser User Agent:", navigator.userAgent);
                reject(new Error(`Failed to load background image from URL: ${imageUrl}`));
            };
            
            img.src = imageUrl;
        });
    }
        
    // Add a function to initialize the background - call this during your app startup
    async initializeBackground(imageUrl) {
        try {
            await this.loadBackgroundImage(imageUrl);
            console.log("Background loaded successfully");
        } catch (error) {
            console.error("Failed to load background:", error);
            // Create a solid color background instead
            this.backgroundTexture = this.device.createTexture({
                size: [this.width, this.height],
                format: this.format,
                usage: 
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.STORAGE_BINDING
            });
            
            // Fill with a solid color (gray)
            const data = new Uint8Array(this.width * this.height * 4);
            for (let i = 0; i < data.length; i += 4) {
                data[i] = 128;     // R
                data[i + 1] = 128; // G
                data[i + 2] = 128; // B
                data[i + 3] = 255; // A
            }
            
            this.device.queue.writeTexture(
                { texture: this.backgroundTexture },
                data,
                { bytesPerRow: this.width * 4 },
                [this.width, this.height]
            );
        }
    }

    async addLayers(sdf, reconstruction, points, depth) {
		const reconstructionTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING,
		});
		const sdfTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING,
		});
		const pointsTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage:
				GPUTextureUsage.TEXTURE_BINDING |
				GPUTextureUsage.COPY_DST |
				GPUTextureUsage.COPY_SRC |
				GPUTextureUsage.STORAGE_BINDING,
		});

		const commandEncoder = this.device.createCommandEncoder();
		commandEncoder.copyTextureToTexture(
			{ texture: reconstruction },
			{ texture: reconstructionTexture },
			[this.width, this.height]
		);

		commandEncoder.copyTextureToTexture(
			{ texture: sdf },
			{ texture: sdfTexture },
			[this.width, this.height]
		);
		commandEncoder.copyTextureToTexture(
			{ texture: points },
			{ texture: pointsTexture },
			[this.width, this.height]
		);
		this.device.queue.submit([commandEncoder.finish()]);

		this.depthPoints.push(pointsTexture);
		this.reconstructions.push(reconstructionTexture);
		this.sdfs.push(sdfTexture);
		this.depths.push(depth);
	}

	async compositeDepths() {
		const reconstructionTextures = this.device.createTexture({
			size: {
				width: this.width,
				height: this.height,
				depthOrArrayLayers: this.reconstructions.length,
			},
			format: "rgba32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		});

		const sdfTextures = this.device.createTexture({
			size: {
				width: this.width,
				height: this.height,
				depthOrArrayLayers: this.sdfs.length,
			},
			format: "rgba32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		});

		const pointsTexture = this.device.createTexture({
			size: {
				width: this.width,
				height: this.height,
				depthOrArrayLayers: this.depthPoints.length,
			},
			format: "rgba32float",
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
		});

		const outputTexture = this.device.createTexture({
			size: [this.width, this.height],
			format: "rgba32float",
			usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
		});

		const uniformData = new ArrayBuffer(12);
		const uniformView = new Uint32Array(uniformData);
		uniformView[0] = this.width;
		uniformView[1] = this.height;
		uniformView[2] = this.reconstructions.length;

		const uniformBuffer = this.device.createBuffer({
			size: 12,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

		const commandEncoder = this.device.createCommandEncoder();

		for (let i = 0; i < this.reconstructions.length; i++) {
			commandEncoder.copyTextureToTexture(
				{ texture: this.reconstructions[i] },
				{
					texture: reconstructionTextures,
					origin: { x: 0, y: 0, z: i },
				},
				[this.width, this.height, 1]
			);
			commandEncoder.copyTextureToTexture(
				{ texture: this.sdfs[i] },
				{ texture: sdfTextures, origin: { x: 0, y: 0, z: i } },
				[this.width, this.height, 1]
			);
			commandEncoder.copyTextureToTexture(
				{ texture: this.depthPoints[i] },
				{ texture: pointsTexture, origin: { x: 0, y: 0, z: i } },
				[this.width, this.height, 1]
			);
		}

		const depthsBuffer = this.device.createBuffer({
			size: this.depthPoints.length  * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});
        this.device.queue.writeBuffer(depthsBuffer, 0, new Float32Array(this.depths))

        const iDepthsBuffer = this.device.createBuffer({
			size: this.depthPoints.length * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

        const iColorXBuffer = this.device.createBuffer({
			size: this.depthPoints.length * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

        const iColorYBuffer = this.device.createBuffer({
			size: this.depthPoints.length * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		})

		const compositePipeline = await this.createCompositePipeline();

		const textureBindGroup = this.device.createBindGroup({
			layout: compositePipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: reconstructionTextures.createView(),
				},
				{
					binding: 1,
					resource: sdfTextures.createView(),
				},
				{
					binding: 2,
					resource: pointsTexture.createView(),
				},
				{
					binding: 3,
					resource: outputTexture.createView(),
				},
			],
		});

		const uniformBindGroup = this.device.createBindGroup({
			layout: compositePipeline.getBindGroupLayout(1),
			entries: [
				{
					binding: 0,
					resource: { buffer: uniformBuffer },
				},
				{
					binding: 1,
					resource: { buffer: depthsBuffer },
				},
				{
					binding: 2,
					resource: { buffer: iDepthsBuffer },
				},
				{
					binding: 3,
					resource: { buffer: iColorXBuffer },
				},
				{
					binding: 4,
					resource: { buffer: iColorYBuffer },
				},
			],
		});

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(compositePipeline);
		computePass.setBindGroup(0, textureBindGroup);
		computePass.setBindGroup(1, uniformBindGroup);
		computePass.dispatchWorkgroups(
			Math.ceil(this.width / 8),
			Math.ceil(this.height / 8)
		);
		computePass.end();

		const outputBuffer = this.device.createBuffer({
			size: this.width * this.height * 4 * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		commandEncoder.copyTextureToBuffer(
			{
				texture: outputTexture,
				mipLevel: 0,
				origin: { x: 0, y: 0, z: 0 },
			},
			{
				buffer: outputBuffer,
				bytesPerRow: this.width * 16,
				rowsPerImage: this.height,
			},
			[this.width, this.height, 1]
		);

		this.device.queue.submit([commandEncoder.finish()]);

		// Map the output buffer to read the data
		await outputBuffer.mapAsync(GPUMapMode.READ);
		const outputData = new Float32Array(outputBuffer.getMappedRange());

        const copiedData = new Float32Array(outputData);

		// saveTextureToPNG(copiedData, this.width, this.height, "output.png");

		// Clean up temporary resources
		outputBuffer.unmap();

		reconstructionTextures.destroy();
		sdfTextures.destroy();
		pointsTexture.destroy();
		outputTexture.destroy();
		uniformBuffer.destroy();

		this.reconstructions = [];
		this.sdfs = [];
		this.depthPoints = [];
		this.depths = [];

        return copiedData;
	}
}