export class Composer {
	constructor(document, canvas, device, format) {
        this.document = document;
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
        this.format = format;
		
        this.pipeline = this.createPipeline();

        this.backgroundTexture = null;
        this.compositeResult = null;
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
                                max(foreground.a, background.a)
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

}