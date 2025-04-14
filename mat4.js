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
