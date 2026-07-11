// Za vsako točko ugotovi njen voksel in atomično poveča glasove za njen top-1 razred.
// Točke z razredom 0 (neklasificirane) so preskočene.

struct Point {
    position:   vec3f,
    color:      u32,
    normal:     vec3f,
    depth:      f32,
    classColor: u32,
    globalIdx:  u32,   // _pad0 — globalni indeks točke pred sortiranjem
    _pad1:      u32,
    _pad2:      u32,
}

// vec4u: gridX, gridY, gridZ, nClasses
// vec4f bbMin, vec4f bbMax
struct VoxelParams {
    grid:  vec4u,
    bbMin: vec4f,
    bbMax: vec4f,
}

@group(0) @binding(0) var<storage, read>            points:          array<Point>;
@group(0) @binding(1) var<uniform>                  nPoints:         u32;

@group(1) @binding(0) var<storage, read>            topK:            array<u32>;
@group(1) @binding(1) var<storage, read_write>      voxelClassVotes: array<atomic<u32>>;
@group(1) @binding(2) var<uniform>                  vp:              VoxelParams;

@compute @workgroup_size(256)
fn voxelVote(@builtin(global_invocation_id) gid: vec3u) {
    let slot = gid.x;
    if slot >= nPoints { return; }

    let globalIdx = points[slot].globalIdx;

    // Top-1 razred za to točko (visoki 16 biti prvega topK zapisa)
    let topClass = topK[globalIdx * 3u] >> 16u;
    if topClass == 0u { return; }  // neklasificirano — preskoči

    // Normaliziramo položaj točke na [0, 1]
    let pos = points[slot].position;
    let fx = clamp((pos.x - vp.bbMin.x) / (vp.bbMax.x - vp.bbMin.x), 0.0, 1.0);
    let fy = clamp((pos.y - vp.bbMin.y) / (vp.bbMax.y - vp.bbMin.y), 0.0, 1.0);
    let fz = clamp((pos.z - vp.bbMin.z) / (vp.bbMax.z - vp.bbMin.z), 0.0, 1.0);

    let vx = min(u32(fx * f32(vp.grid.x)), vp.grid.x - 1u);
    let vy = min(u32(fy * f32(vp.grid.y)), vp.grid.y - 1u);
    let vz = min(u32(fz * f32(vp.grid.z)), vp.grid.z - 1u);

    let voxelIdx = vz * vp.grid.x * vp.grid.y + vy * vp.grid.x + vx;
    atomicAdd(&voxelClassVotes[voxelIdx * vp.grid.w + topClass], 1u);
}
