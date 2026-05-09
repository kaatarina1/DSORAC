export function zero() {
	return new Float32Array(16);
}

export function identity() {
	const out = zero();
	out[0] = 1;
	out[5] = 1;
	out[10] = 1;
	out[15] = 1;
	return out;
}

export function translation(t) {
	const out = identity();
	out[12] = t[0];
	out[13] = t[1];
	out[14] = t[2];
	return out;
}

export function scale(s) {
	const out = identity();
	out[0] = s[0];
	out[5] = s[1];
	out[10] = s[2];
	return out;
}

export function rotateX(a) {
	const out = identity();
	const c = Math.cos(a);
	const s = Math.sin(a);
	out[5] = c;
	out[6] = s;
	out[9] = -s;
	out[10] = c;
	return out;
}

export function rotateY(a) {
	const out = identity();
	const c = Math.cos(a);
	const s = Math.sin(a);
	out[0] = c;
	out[2] = -s;
	out[8] = s;
	out[10] = c;
	return out;
}

export function rotateZ(a) {
	const out = identity();
	const c = Math.cos(a);
	const s = Math.sin(a);
	out[0] = c;
	out[1] = s;
	out[4] = -s;
	out[5] = c;
	return out;
}

export function perspective(fovy, aspect, near, far) {
	const out = zero();
	const f = 1 / Math.tan(fovy / 2);
	out[0] = f / aspect;
	out[5] = f;
	out[10] = far / (near - far);
	out[11] = -1;
	out[14] = (far * near) / (near - far);
	return out;
}

export function inversePerspective(fovy, aspect, near, far) {
	const out = zero();
	const f = 1 / Math.tan(fovy / 2);
	out[0] = aspect / f;
	out[5] = 1 / f;
	out[11] = (near - far) / (far * near);
	out[14] = -1;
	out[15] = 1 / near;
	return out;
}

export function multiplyTwo(a, b) {
	const out = zero();
	for (let j = 0; j < 4; j++) {
		for (let i = 0; i < 4; i++) {
			out[j + i * 4] = 0;
			for (let k = 0; k < 4; k++) {
				out[j + i * 4] += a[j + k * 4] * b[k + i * 4];
			}
		}
	}
	return out;
}

export function multiply(...matrices) {
	return matrices.reduce((a, b) => multiplyTwo(a, b), identity());
}

export function lookAt(eye, target, up) {
    const zAxis = normalize([
        eye[0] - target[0],
        eye[1] - target[1],
        eye[2] - target[2],
    ]);
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = cross(zAxis, xAxis);

    return [
        xAxis[0], yAxis[0], zAxis[0], 0,
        xAxis[1], yAxis[1], zAxis[1], 0,
        xAxis[2], yAxis[2], zAxis[2], 0,
        -dot(xAxis, eye), -dot(yAxis, eye), -dot(zAxis, eye), 1,
    ];
}

export function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function normalize(v) {
    const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    return len > 0 ? [v[0]/len, v[1]/len, v[2]/len] : [0,0,0];
}

export function getIntrinsics(width, height, fovy) {
    const aspect = width / height;

    // focal lengths (COLMAP convention: pixels)
    const fy = (height / 2) / Math.tan(fovy / 2);
    const fx = fy * aspect;

    // principal point (image center)
    const cx = width / 2.0;
    const cy = height / 2.0;

    return { fx, fy, cx, cy, width, height };
}

// function rotationMatrixToQuaternion(R) {
//     // Row-major layout: [r00, r01, r02, r10, r11, r12, r20, r21, r22]
//   const r00 = R[0], r01 = R[1], r02 = R[2];
//   const r10 = R[3], r11 = R[4], r12 = R[5];
//   const r20 = R[6], r21 = R[7], r22 = R[8];

//   const trace = r00 + r11 + r22;
//   let w, x, y, z;

//   if (trace > 0.0) {
//     const S = Math.sqrt(trace + 1.0) * 2.0; // S=4*w
//     w = 0.25 * S;
//     x = (r21 - r12) / S;
//     y = (r02 - r20) / S;
//     z = (r10 - r01) / S;
//   } else if ((r00 > r11) && (r00 > r22)) {
//     const S = Math.sqrt(1.0 + r00 - r11 - r22) * 2.0; // S=4*x
//     w = (r21 - r12) / S;
//     x = 0.25 * S;
//     y = (r01 + r10) / S;
//     z = (r02 + r20) / S;
//   } else if (r11 > r22) {
//     const S = Math.sqrt(1.0 + r11 - r00 - r22) * 2.0; // S=4*y
//     w = (r02 - r20) / S;
//     x = (r01 + r10) / S;
//     y = 0.25 * S;
//     z = (r12 + r21) / S;
//   } else {
//     const S = Math.sqrt(1.0 + r22 - r00 - r11) * 2.0; // S=4*z
//     w = (r10 - r01) / S;
//     x = (r02 + r20) / S;
//     y = (r12 + r21) / S;
//     z = 0.25 * S;
//   }

function quaternionToRotationMatrix(q) {
    const [w, x, y, z] = q;
    const xx = x*x, yy = y*y, zz = z*z;
    const xy = x*y, xz = x*z, yz = y*z;
    const wx = w*x, wy = w*y, wz = w*z;
    
    return [
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ];
}

// Matrix-vector multiplication (3x3 matrix, 3D vector)
function multiplyMatrixVector3(m, v) {
    return [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2]
    ];
}

// Quaternion multiplication: q1 * q2
function multiplyQuaternions(q1, q2) {
    const [w1, x1, y1, z1] = q1;
    const [w2, x2, y2, z2] = q2;
    
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2, // w
        w1*x2 + x1*w2 + y1*z2 - z1*y2, // x
        w1*y2 - x1*z2 + y1*w2 + z1*x2, // y
        w1*z2 + x1*y2 - y1*x2 + z1*w2  // z
    ];
}

// Convert rotation matrix to quaternion (for 3x3 matrix in row-major format)
function rotationMatrixToQuaternion(R) {
    // R is 3x3 matrix in array format: [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    const r00 = R[0][0], r01 = R[0][1], r02 = R[0][2];
    const r10 = R[1][0], r11 = R[1][1], r12 = R[1][2];
    const r20 = R[2][0], r21 = R[2][1], r22 = R[2][2];
    
    const trace = r00 + r11 + r22;
    let w, x, y, z;
    
    if (trace > 0.0) {
        const S = Math.sqrt(trace + 1.0) * 2.0; // S=4*w
        w = 0.25 * S;
        x = (r21 - r12) / S;
        y = (r02 - r20) / S;
        z = (r10 - r01) / S;
    } else if ((r00 > r11) && (r00 > r22)) {
        const S = Math.sqrt(1.0 + r00 - r11 - r22) * 2.0; // S=4*x
        w = (r21 - r12) / S;
        x = 0.25 * S;
        y = (r01 + r10) / S;
        z = (r02 + r20) / S;
    } else if (r11 > r22) {
        const S = Math.sqrt(1.0 + r11 - r00 - r22) * 2.0; // S=4*y
        w = (r02 - r20) / S;
        x = (r01 + r10) / S;
        y = 0.25 * S;
        z = (r12 + r21) / S;
    } else {
        const S = Math.sqrt(1.0 + r22 - r00 - r11) * 2.0; // S=4*z
        w = (r10 - r01) / S;
        x = (r02 + r20) / S;
        y = (r12 + r21) / S;
        z = 0.25 * S;
    }
    
    // Normalize
    const norm = Math.sqrt(w*w + x*x + y*y + z*z);
    if (norm === 0) return [1, 0, 0, 0];
    return [w/norm, x/norm, y/norm, z/norm]; // [w, x, y, z]
}

// FIXED: Simple and direct conversion using camera position and target
export function cameraToColmapPose(cameraPos, viewMatrix) {
    // Use the provided camera position (from your hemisphere generation)
    // Don't extract from view matrix since your generation is correct
    
    // Extract the rotation part from view matrix (upper-left 3x3)
    // View matrix contains R^T, so we need to transpose it
    const R_camera = [
        [viewMatrix[0], viewMatrix[4], viewMatrix[8]],   // Transpose to get camera-to-world
        [viewMatrix[1], viewMatrix[5], viewMatrix[9]],
        [viewMatrix[2], viewMatrix[6], viewMatrix[10]]
    ];
    
    // Convert rotation matrix to quaternion
    let quat_webgl = rotationMatrixToQuaternion(R_camera);
    
    // Convert from WebGL coordinate system to COLMAP coordinate system
    // WebGL: Y-up, -Z forward
    // COLMAP: Y-down, +Z forward
    // Apply 180° rotation around X-axis
    const fixup = [0, 1, 0, 0]; // 180° around X-axis [w,x,y,z]
    const quat_colmap = multiplyQuaternions(fixup, quat_webgl);
    
    // Convert quaternion to rotation matrix for translation calculation
    const R_colmap = quaternionToRotationMatrix(quat_colmap);
    
    // COLMAP translation: t = -R * camera_position
    const t_temp = multiplyMatrixVector3(R_colmap, cameraPos);
    const translation = [-t_temp[0], -t_temp[1], -t_temp[2]];
    
    return { 
        quat: quat_colmap,  // [w, x, y, z] format for COLMAP
        t: translation 
    };
}

// Alternative: Direct conversion from camera position and target (bypassing view matrix issues)
export function cameraToColmapPoseFromTarget(cameraPos, target, up = [0, 1, 0]) {
    // Create camera-to-world rotation matrix directly
    const forward = normalize([
        target[0] - cameraPos[0],
        target[1] - cameraPos[1], 
        target[2] - cameraPos[2]
    ]);
    
    const right = normalize(cross(up, forward));
    const newUp = cross(forward, right);
    
    // Camera-to-world matrix (inverse of lookAt)
    const R_camera = [
        [right[0], right[1], right[2]],
        [newUp[0], newUp[1], newUp[2]], 
        [-forward[0], -forward[1], -forward[2]]  // -forward for camera looking direction
    ];
    
    // Convert to quaternion
    let quat_webgl = rotationMatrixToQuaternion(R_camera);
    
    // Apply COLMAP coordinate system conversion (180° around X)
    const fixup = [0, 1, 0, 0];
    const quat_colmap = multiplyQuaternions(fixup, quat_webgl);
    
    // Calculate COLMAP translation
    const R_colmap = quaternionToRotationMatrix(quat_colmap);
    const t_temp = multiplyMatrixVector3(R_colmap, cameraPos);
    const translation = [-t_temp[0], -t_temp[1], -t_temp[2]];
    
    return {
        quat: quat_colmap,
        t: translation
    };
}

export function orthographic(left, right, bottom, top, near, far) {
    const out = zero();
    out[0]  =  2 / (right - left);
    out[5]  =  2 / (top - bottom);
    out[10] = -2 / (far - near);
    out[12] = -(right + left) / (right - left);
    out[13] = -(top + bottom) / (top - bottom);
    out[14] = -(far + near)   / (far - near);
    out[15] =  1;
    return out;
}