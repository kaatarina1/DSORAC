import * as mat4 from "./mat4.js";

export class CameraPosition {
	constructor(canvas, device, renderingPipeline) {
		this.canvas = canvas;
		this.device = device;
		this.width = canvas.width;
		this.height = canvas.height;
        this.renderingPipeline = renderingPipeline;
	}

generateFibonacciHemisphereAroundCamera(samples, radius, target) {
    const goldenAngle = Math.PI * (3.0 - Math.sqrt(5.0));
    const poses = [];

    for (let i = 0; i < samples; i++) {
        const y = 1.0 - (i / (samples - 1)) * 2.0; // from 1 → -1
        if (y < 0) continue; // Only take the upper hemisphere (y >= 0)
        const r = Math.sqrt(1.0 - y * y);

        const theta = goldenAngle * i;

        const x = r * Math.cos(theta);
        const z = r * Math.sin(theta);

        // Hemisphere should point downward (negative Y)
        const pos = [
            target[0] + radius * x,
            target[1] + radius * y, // This is the "down" direction
            target[2] + radius * z,
        ];

        poses.push(pos);
    }
    return poses;
}

generateTest(samples, radius, target) {
    const poses = [];
    const heightAboveTarget = 0.5; // Adjust this value for desired height offset

    for (let i = 0; i < samples; i++) {
        // Calculate angle for each sample (evenly distributed around circle)
        const angle = (i / samples) * 2 * Math.PI;
        
        const x = target[0];
        const y = target[1] + radius ;
        const z = target[2];

        poses.push([x, y, z]);
    }
    return poses;
}

computeCameraMatrix(eye, target, canvas) {
    // For Y-up system looking downward
    const dir = [
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2]
    ];
    const dirNorm = Math.hypot(...dir);
    const dirUnit = dir.map(v => v / dirNorm);

    // Default up vector is Y-up
    let up = [0, 1, 0];
    
    // If looking almost straight down (Y direction), adjust up vector
    if (Math.abs(dirUnit[1]) > 0.99) {
        up = [0, 0, 1]; // Use Z as up when looking mostly downward
    }

    const viewMatrix = mat4.lookAt(eye, target, up);
    const projectionMatrix = mat4.perspective(
        45 * Math.PI / 180,
        canvas.width / canvas.height,
        0.1,
        2000
    );
    const projectionViewMatrix = mat4.multiply(projectionMatrix, viewMatrix);
    return {viewMatrix, projectionViewMatrix};
}
    // Create a temporary matrix buffer + bind group for an arbitrary 4x4 matrix (Float32Array or Array[16])
    createTempMatrixBindGroup(matrixArray) {
        // ensure Float32Array
        const mat = (matrixArray instanceof Float32Array) ? matrixArray : new Float32Array(matrixArray);

        const tempMatrixBuffer = this.device.createBuffer({
            size: 64, // 16 floats * 4 bytes
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // upload data
        this.device.queue.writeBuffer(tempMatrixBuffer, 0, mat);

        const tempBindGroup = this.device.createBindGroup({
            layout: this.renderingPipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: tempMatrixBuffer },
                },
            ],
        });

        return { tempMatrixBuffer, tempBindGroup };
    }

    writeCamerasTxt(captured) {
        let cam = captured[0];
        return `# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        1 PINHOLE ${cam.width} ${cam.height} ${cam.fx} ${cam.fy} ${cam.cx} ${cam.cy}
        `;
    }

    writeImagesTxt(captured, scaleFactor=1.0) {
        let lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"];
        for (let c of captured) {
            lines.push(`${c.imageId} ${c.quat[0]} ${c.quat[1]} ${c.quat[2]} ${c.quat[3]} ${c.t[0] * scaleFactor} ${c.t[1] * scaleFactor} ${c.t[2] * scaleFactor} 1 ${c.filename}\n`);
        }
        return lines.join("\n");
    }

    writePoints3DTxt(positions, colors, scaleFactor=1.0) {
        const points = [];
        for (let i = 0; i < positions.length; i+=200) {
            points.push({
                x: positions[i] * scaleFactor,
                y: positions[i+1] * scaleFactor,
                z: positions[i+2] * scaleFactor,
                r: Math.round(colors[i] * 255),
                g: Math.round(colors[i+1] * 255),
                b: Math.round(colors[i+2] * 255),
            });
        }
        return `# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n` +
            points.map((p,i)=>`${i} ${p.x} ${p.y} ${p.z} ${p.r} ${p.g} ${p.b} 0`).join("\n");
    }
}
