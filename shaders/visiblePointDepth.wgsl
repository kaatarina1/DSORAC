struct Point {
    position:      vec3f,
    color:         u32,
    normal:        vec3f,
    depth:         f32,
    classColor:    u32,
    _pad0:         u32,
    lasClassColor: u32,
    _pad2:         u32,
}

// Vsak chunk (pointcloud buffer) ima svoje število točk in odmik v skupnem izhodnem bufferju.
struct ChunkParams {
    count:  u32,
    offset: u32,
}

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<uniform> chunk: ChunkParams;
@group(0) @binding(2) var<storage, read_write> outputDepths: array<f32>;
@group(1) @binding(0) var<uniform> projectionViewMatrix: mat4x4f;
@group(1) @binding(1) var<uniform> viewMatrix: mat4x4f;
@group(1) @binding(2) var<uniform> isSpherical: f32;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    // Buffer je alociran na maxPointsPerBuffer, zato ne smemo brati čez dejansko število točk.
    if (index >= chunk.count) {
        return;
    }

    let pos = points[index].position;

    if (isSpherical > 0.5) {
        // Sferičen (panoramski) zajem nima prave frustum projekcije - vsaka
        // točka je v neki smeri "vidna", zato brez NDC/frustum testa računamo
        // le radialno razdaljo od kamere (usklajeno z rendering.wgsl).
        let viewPos = viewMatrix * vec4f(pos, 1.0);
        outputDepths[chunk.offset + index] = length(viewPos.xyz);
    } else {
        // preverimo ali je znotraj viewa
        let clipPos = projectionViewMatrix * vec4f(pos, 1.0);
        let inView = clipPos.w > 0.0
            && abs(clipPos.x) <= clipPos.w
            && abs(clipPos.y) <= clipPos.w
            && clipPos.z >= 0.0 && clipPos.z <= clipPos.w;

        if (inView) {
            let viewPos = viewMatrix * vec4f(pos, 1.0);
            outputDepths[chunk.offset + index] = -viewPos.z; // linearna globina v prostoru pogleda
        } else {
            outputDepths[chunk.offset + index] = -1.0;
        }
    }
}
