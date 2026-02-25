export class DepthMap {
	constructor(canvas, device, depthTexture) {
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
		this.depthTexture = depthTexture;

		this.depthStorageBuffer = this.device.createBuffer({
			size: this.width * this.height * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});

		this.readbackBuffer = this.device.createBuffer({
			size: this.width * this.height * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});

		this.computeShader = device.createShaderModule({
			code: `
              @group(0) @binding(0) var depthTexture: texture_depth_2d;
              @group(0) @binding(1) var<storage, read_write> outputDepths: array<f32>;
              
              @compute @workgroup_size(16, 16)
              fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let dimensions = textureDimensions(depthTexture);
                if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
                  return;
                }
                
                let texCoord = vec2i(global_id.xy);
                let depth = textureLoad(depthTexture, texCoord, 0);
                
                let index = global_id.y * dimensions.x + global_id.x;
                outputDepths[index] = depth;
              }
            `,
		});

		this.computePipeline = this.device.createComputePipeline({
			layout: "auto",
			compute: {
				module: this.computeShader,
				entryPoint: "main",
			},
		});

		this.computeBindGroup = this.device.createBindGroup({
			layout: this.computePipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: this.depthTexture.createView({
						format: "depth32float", 
						aspect: "depth-only", 
					}),
				},
				{
					binding: 1,
					resource: {
						buffer: this.depthStorageBuffer,
					},
				},
			],
		});
	}

	// Funkcija za ekstrakcijo globinskih vrednosti iz depth texture in grupiranje v bin-e.
	async extractDepthValues() {
		const commandEncoder = this.device.createCommandEncoder();

		const computePass = commandEncoder.beginComputePass();
		computePass.setPipeline(this.computePipeline);
		computePass.setBindGroup(0, this.computeBindGroup);

		const workgroupCountX = Math.ceil(this.width / 16);
		const workgroupCountY = Math.ceil(this.height / 16);

		computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
		computePass.end();

		commandEncoder.copyBufferToBuffer(
			this.depthStorageBuffer,
			0,
			this.readbackBuffer,
			0,
			this.width * this.height * 4
		);

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

		await this.readbackBuffer.mapAsync(GPUMapMode.READ);
		const arrayBuffer = this.readbackBuffer.getMappedRange();
		const depthValues = new Float32Array(arrayBuffer.slice(0));

		this.readbackBuffer.unmap();

		return depthValues;
	}

	/**
	 * Deljenje globine v bin-e, ki vsebujejo približno enako število točk, ne glede na razdaljo kamere. 
	 * Bin-i so v linearnih enotah pogleda, da jih shader lahko neposredno primerja.
	 */
	async groupDepthIntoBins({
		numBins = 12, // Število ciljanih binov (globinskih rezin)
		near = 0.05, // Near ravnina (mora biti usklajena s tisto, ki jo shader uporablja za renderiranje)
		far = 5.0, // Far ravnina (mora biti usklajena s tisto, ki jo shader uporablja za renderiranje)
		minPointsFraction = 0.01, // Minimalni delež točk v binu, preden se združi
		minBinWidth = 0.002, // Minimalna širina bin-a v linearni razdalji
	} = {}) {
		const depthValues = await this.extractDepthValues();

		// Lineariziraj globinske vrednosti iz NDC v linearne enote pogleda
		const linearize = (d) => {
			const z_ndc = d * 2.0 - 1.0;
			return (2.0 * near * far) / (far + near - z_ndc * (far - near));
		};

		const linear = [];
		for (let i = 0; i < depthValues.length; i++) {
			const d = depthValues[i];
			// Zavrnemo vrednosti, ki so točno 1.0 (neveljavno/ nebo) ali 0.0 (morda ekstremno blizu, lahko tudi artefakti), 
			// ter tiste, ki so izven uporabnega razpona.
			if (d > 0.0 && d < 0.999) {
				const lv = linearize(d);
				if (Number.isFinite(lv) && lv > 0) {
					linear.push(lv);
				}
			}
		}

		if (linear.length === 0) {
			console.warn("No valid depth samples — returning single fallback bin");
			return [[near, far]];
		}

		// Sortiraj linearne globinske vrednosti, da lahko zgradimo kvantilne bin-e.
		linear.sort((a, b) => a - b);

		const totalPoints = linear.length;
		const minPointsPerBin = Math.max(1, Math.floor(totalPoints * minPointsFraction));

		//Ustvari surove bin-e na osnovi kvantilov, 
		// da zagotovimo približno enako število točk v vsakem binu, ne glede na razdaljo.
		let rawBins = [];
		for (let i = 0; i < numBins; i++) {
			const startIdx = Math.floor((i / numBins) * totalPoints);
			const endIdx = Math.min(
				Math.floor(((i + 1) / numBins) * totalPoints) - 1,
				totalPoints - 1
			);
			const start = linear[startIdx];
			const end = linear[endIdx];
			rawBins.push({
				start,
				end,
				count: endIdx - startIdx + 1,
			});
		}

		// Ydružimo surove bin-e, ki imajo premalo točk ali so preozki, da zagotovimo stabilnost rekonstrukcije.
		let merged = [rawBins[0]];
		for (let i = 1; i < rawBins.length; i++) {
			const prev = merged[merged.length - 1];
			const cur = rawBins[i];
			const width = cur.end - cur.start;
			if (width < minBinWidth || cur.count < minPointsPerBin) {
				prev.end = cur.end;
				prev.count += cur.count;
			} else {
				merged.push(cur);
			}
		}


		// Razširimo prvega/ zadnjega bina na celoten near/far razpon, da nobena točka ne pade izven pokritosti.
		const bins = merged.map((b, i) => {
			let s = b.start;
			let e = b.end;
			// Prvi bin se razteza vse do near, da zajame vse točke blizu kamere
			if (i === 0) s = Math.max(0, near * 0.5);
			// Zadnji bin se razteza vse do far, da zajame vse točke daleč od kamere
			if (i === merged.length - 1) e = far * 1.5;
			return [s, e];
		});

		// Naredimo bin-e povezane (continous) in dodamo prekrivanje.
		// Prekrivanje zagotavlja, da točke blizu meje pojavijo v obeh
		// sosednjih rezinah, kar daje rekonstrukcijskemu solverju dovolj podatkov.
		for (let i = 1; i < bins.length; i++) {
			const mid = (bins[i - 1][1] + bins[i][0]) / 2;
			const binWidth = bins[i][1] - bins[i][0];
			const prevBinWidth = bins[i - 1][1] - bins[i - 1][0];
			const overlap = Math.min(binWidth, prevBinWidth) * 0.15;
			bins[i - 1][1] = mid + overlap;
			bins[i][0] = mid - overlap;
		}

		console.log(
			`Depth binning: ${totalPoints} valid points → ${bins.length} bins (linear view-space)`,
			bins
		);

		return bins;
	}
}