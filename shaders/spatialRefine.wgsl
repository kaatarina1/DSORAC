// Prostorsko glasovanje: za vsako točko upošteva top-3 kandidate in jih pretehtuje
// s podporo v 3×3×3 sosednjih vokslih.
//
// Score razreda k = (glasovi_k / skupaj_glasov) × (število_sosednjih_vokslov_z_razredom_k + 1)
// "+1" zagotavlja, da razred z dobro maskno podporo ne dobi score 0, ko ni sosednje podpore.

struct Point {
    position:   vec3f,
    color:      u32,
    normal:     vec3f,
    depth:      f32,
    classColor: u32,
    globalIdx:  u32,
    _pad1:      u32,
    _pad2:      u32,
}

struct VoxelParams {
    grid:  vec4u,  // gridX, gridY, gridZ, unused
    bbMin: vec4f,
    bbMax: vec4f,
}

@group(0) @binding(0) var<storage, read>       points:      array<Point>;
@group(0) @binding(1) var<uniform>             nPoints:     u32;

@group(1) @binding(0) var<storage, read>       topK:        array<u32>;
@group(1) @binding(1) var<storage, read>       voxelWinner: array<u32>;
@group(1) @binding(2) var<storage, read>       palette:     array<u32>;
@group(1) @binding(3) var<storage, read_write> winner:      array<u32>;
@group(1) @binding(4) var<uniform>             vp:          VoxelParams;

@compute @workgroup_size(256)
fn spatialRefine(@builtin(global_invocation_id) gid: vec3u) {
    let slot = gid.x;
    if slot >= nPoints { return; }

    let globalIdx = points[slot].globalIdx;

    // Preberi top-3 kandidate: (classId << 16) | count
    let tk0 = topK[globalIdx * 3u + 0u];
    let tk1 = topK[globalIdx * 3u + 1u];
    let tk2 = topK[globalIdx * 3u + 2u];

    let class0 = tk0 >> 16u;
    let class1 = tk1 >> 16u;
    let class2 = tk2 >> 16u;
    let votes0  = f32(tk0 & 0xFFFFu);
    let votes1  = f32(tk1 & 0xFFFFu);
    let votes2  = f32(tk2 & 0xFFFFu);

    // Voksel koordinate te točke
    let pos = points[slot].position;
    let fx = clamp((pos.x - vp.bbMin.x) / (vp.bbMax.x - vp.bbMin.x), 0.0, 1.0);
    let fy = clamp((pos.y - vp.bbMin.y) / (vp.bbMax.y - vp.bbMin.y), 0.0, 1.0);
    let fz = clamp((pos.z - vp.bbMin.z) / (vp.bbMax.z - vp.bbMin.z), 0.0, 1.0);

    let vx = i32(min(u32(fx * f32(vp.grid.x)), vp.grid.x - 1u));
    let vy = i32(min(u32(fy * f32(vp.grid.y)), vp.grid.y - 1u));
    let vz = i32(min(u32(fz * f32(vp.grid.z)), vp.grid.z - 1u));

    // Preštej koliko od 27 sosednjih vokslov podpira vsak top-3 kandidat
    var spatial0 = 0u;
    var spatial1 = 0u;
    var spatial2 = 0u;

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let nx = vx + dx;
                let ny = vy + dy;
                let nz = vz + dz;
                if nx < 0 || ny < 0 || nz < 0 ||
                   nx >= i32(vp.grid.x) || ny >= i32(vp.grid.y) || nz >= i32(vp.grid.z) {
                    continue;
                }
                let nvox = u32(nz) * vp.grid.x * vp.grid.y + u32(ny) * vp.grid.x + u32(nx);
                let nw   = voxelWinner[nvox];
                if nw == class0 { spatial0 += 1u; }
                if nw == class1 { spatial1 += 1u; }
                if nw == class2 { spatial2 += 1u; }
            }
        }
    }

    // Score = relativni delež glasov × prostorska podpora
    // "+1" ohrani signal za razred brez sosednje podpore
    let totalVotes = votes0 + votes1 + votes2 + 1.0;
    let score0 = (votes0 / totalVotes) * f32(spatial0 + 1u);
    let score1 = (votes1 / totalVotes) * f32(spatial1 + 1u);
    let score2 = (votes2 / totalVotes) * f32(spatial2 + 1u);

    var bestClass = class0;
    var bestScore = score0;
    if score1 > bestScore { bestScore = score1; bestClass = class1; }
    if score2 > bestScore { bestScore = score2; bestClass = class2; }

    winner[globalIdx] = palette[bestClass & 0xFFu];
}
