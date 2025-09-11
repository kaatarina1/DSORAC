import { saveMaskToPNG, saveSDFToPNG, saveTextureToPNG } from "./Utils";

export class SignedDistanceFiled {
  constructor(device, texture, width, height) {
    this.device = device;
    this.texture = texture;
    this.width = width;
    this.height = height;

    this.closedPipeline = null;
    this.initCoordsPipeline = null;
    this.jfaPipeline = null;
    this.distancePipeline = null;
    this.blurPipeline = null;

    this.coordATexture = null;
    this.coordBTexture = null;
    this.sdfTexture = null;
    this.sdfBlurred = null;

    this.jfaParamsBuffer = null;   // step (i32), width (u32), height (u32)
    this.distParamsBuffer = null;  // normFactor (f32), width (f32), height (f32)
  }

  getJfaParamsBuffer() {
    if (!this.jfaParamsBuffer) {
      this.jfaParamsBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    return this.jfaParamsBuffer;
  }

  getDistParamsBuffer() {
    if (!this.distParamsBuffer) {
      this.distParamsBuffer = this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    return this.distParamsBuffer;
  }

  async createClosedPipeline() {
    if (this.closedPipeline) return this.closedPipeline;
    const closedCode = await fetch("shaders/isColored.wgsl").then(r => r.text());
    const module = this.device.createShaderModule({ code: closedCode });
    this.closedPipeline = this.device.createComputePipeline({ 
      layout: "auto", 
      compute: { 
        module, 
        entryPoint: "main" 
      } 
    });
    return this.closedPipeline;
  }

  async createInitCoordsPipeline() {
    if (this.initCoordsPipeline) return this.initCoordsPipeline;
    const code = await fetch("shaders/initCoords.wgsl").then(r => r.text());
    const module = this.device.createShaderModule({ code });
    this.initCoordsPipeline = this.device.createComputePipeline({ 
      layout: "auto", 
      compute: { 
        module, 
        entryPoint: "main" 
      } 
    });
    return this.initCoordsPipeline;
  }

  async createJfaPipeline() {
    if (this.jfaPipeline) return this.jfaPipeline;
    const code = await fetch("shaders/jfaStep.wgsl").then(r => r.text());
    const module = this.device.createShaderModule({ code });
    this.jfaPipeline = this.device.createComputePipeline({ 
      layout: "auto", 
      compute: { 
        module, 
        entryPoint: "main" 
      } 
    });
    return this.jfaPipeline;
  }

  async createDistancePipeline() {
    if (this.distancePipeline) return this.distancePipeline;
    const code = await fetch("shaders/jfaDistance.wgsl").then(r => r.text());
    const module = this.device.createShaderModule({ code });
    this.distancePipeline = this.device.createComputePipeline({ 
      layout: "auto", 
      compute: { 
        module, 
        entryPoint: "main" 
      } 
    });
    return this.distancePipeline;
  }

  async createBlurPipeline() {
    if (this.blurPipeline) return this.blurPipeline;
    const code = await fetch("shaders/jfaBlur.wgsl").then(r => r.text());
    const module = this.device.createShaderModule({ code });
    this.blurPipeline = this.device.createComputePipeline({ 
      layout: "auto", 
      compute: { 
        module, 
        entryPoint: "main" 
      } 
    });
    return this.blurPipeline;
  }

  createMaskTexture() {
    return this.device.createTexture({ 
      size: [this.width, this.height], 
      format: "r32uint", 
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC 
    });
  }

  createCoordTexture() {
    return this.device.createTexture({ 
      size: [this.width, this.height], 
      format: "rgba32sint", 
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST 
    });
  }

  createSdfTexture() {
    return this.device.createTexture({ 
      size: [this.width, this.height], 
      format: "rgba32float", 
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST 
    });
  }

  highestPow2LE(n) { 
    let p = 1; 
    while ((p << 1) <= n) p <<= 1; 
    return p; 
  }

  async generateSDF() {
    await this.createClosedPipeline();
    await this.createInitCoordsPipeline();
    await this.createJfaPipeline();
    await this.createDistancePipeline();
    await this.createBlurPipeline();

    const maskTex = this.createMaskTexture();
    this.coordATexture = this.createCoordTexture();
    this.coordBTexture = this.createCoordTexture();
    this.sdfTexture = this.createSdfTexture();
    this.sdfBlurred = this.createSdfTexture();

    // Seed mask
    const maskBind = this.device.createBindGroup({ 
      layout: this.closedPipeline.getBindGroupLayout(0), 
      entries: [ 
        { 
          binding: 0, resource: this.texture.createView() 
        }, 
        { 
          binding: 1, resource: maskTex.createView() 
        } 
      ] 
    });
    const enc0 = this.device.createCommandEncoder();
    { 
      const pass = enc0.beginComputePass(); 
      pass.setPipeline(this.closedPipeline); 
      pass.setBindGroup(0, maskBind); 
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8), 1); 
      pass.end(); 
    }

    // Init coords from mask
    const initBind = this.device.createBindGroup({ 
      layout: this.initCoordsPipeline.getBindGroupLayout(0), 
      entries: [ 
        { 
          binding: 0, resource: maskTex.createView() 
        }, 
        { 
          binding: 1, resource: this.coordATexture.createView() 
        } 
      ] 
    });
    { 
      const pass = enc0.beginComputePass(); 
      pass.setPipeline(this.initCoordsPipeline); 
      pass.setBindGroup(0, initBind); 
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8), 1); 
      pass.end(); 
    }

    this.device.queue.submit([enc0.finish()]);

    // JFA passes (ping-pong)
    const maxDim = Math.max(this.width, this.height);
    let step = this.highestPow2LE(maxDim);
    if (step > maxDim) step >>= 1;

    const jfaParams = this.getJfaParamsBuffer();
    const doJfaPass = (inputTex, outputTex, stepVal) => {
      const u32 = new Uint32Array([stepVal >>> 0, this.width >>> 0, this.height >>> 0, 0]);
      this.device.queue.writeBuffer(jfaParams, 0, u32.buffer);
      const bind = this.device.createBindGroup({ 
        layout: this.jfaPipeline.getBindGroupLayout(0), 
        entries: [ 
          { 
            binding: 0, resource: inputTex.createView() 
          }, 
          { 
            binding: 1, resource: outputTex.createView()             
          }, 
          { 
            binding: 2, resource: { buffer: jfaParams } 
          } 
        ] 
      });
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass(); 
      pass.setPipeline(this.jfaPipeline); 
      pass.setBindGroup(0, bind); 
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8), 1); 
      pass.end();
      
      this.device.queue.submit([enc.finish()]);
    };


    let pingIn = this.coordATexture;
    let pingOut = this.coordBTexture;
    while (step >= 1) { 
      // // After the mask generation pass
      // const buffer = await this.device.createBuffer({
      //   size: this.width * this.height * 16,
      //   usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      // });

      // const encTest = this.device.createCommandEncoder();
      // encTest.copyTextureToBuffer(
      //   { texture: pingIn },
      //   { buffer, bytesPerRow: this.width * 16 },
      //   [this.width, this.height]
      // );
      // this.device.queue.submit([encTest.finish()]);

      // await buffer.mapAsync(GPUMapMode.READ);
      // const data = new Uint32Array(buffer.getMappedRange());
      
      // saveTextureToPNG(data, this.width, this.height, "mask.png");
      // buffer.unmap();

      doJfaPass(pingIn, pingOut, step); 
      [pingIn, pingOut] = [pingOut, pingIn];
      step >>= 1; 
    }

    const diag = Math.hypot(this.width, this.height);
    const norm = 1.0 / Math.max(1.0, diag);
    const f32 = new Float32Array([norm, this.width, this.height, 0.0]);
    const distParams = this.getDistParamsBuffer();
    this.device.queue.writeBuffer(distParams, 0, f32.buffer);

    const distBind = this.device.createBindGroup({ 
      layout: this.distancePipeline.getBindGroupLayout(0), 
      entries: [ 
        { 
          binding: 0, resource: pingIn.createView() 
        }, 
        { 
          binding: 1, resource: this.sdfTexture.createView() 
        }, 
        { 
          binding: 2, resource: { buffer: distParams } 
        } 
      ] 
    });

    const enc2 = this.device.createCommandEncoder();
    { 
      const pass = enc2.beginComputePass(); 
      pass.setPipeline(this.distancePipeline); 
      pass.setBindGroup(0, distBind); 
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8), 1); 
      pass.end(); 
    }
    this.device.queue.submit([enc2.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // Blur pass
    const blurBind = this.device.createBindGroup({ 
      layout: this.blurPipeline.getBindGroupLayout(0), 
      entries: [ 
        { 
          binding: 0, resource: this.sdfTexture.createView() 
        }, 
        { 
          binding: 1, resource: this.sdfBlurred.createView() 
        } 
      ] 
    });

    const enc3 = this.device.createCommandEncoder();
    { 
      const pass = enc3.beginComputePass(); 
      pass.setPipeline(this.blurPipeline); 
      pass.setBindGroup(0, blurBind); 
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8), 1); 
      pass.end(); 
    }
    this.device.queue.submit([enc3.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    return this.sdfBlurred;
  }
}