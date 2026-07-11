// Za vsak voksel poišče razred z največ glasovi → voxelWinner[v] = dominantni classId.
// Prazni voksli (brez točk) dobijo winner = 0 (neklasificirano).

@group(0) @binding(0) var<storage, read>       voxelClassVotes: array<u32>;
@group(0) @binding(1) var<storage, read_write> voxelWinner:     array<u32>;
@group(0) @binding(2) var<uniform>             params:          vec2u;  // nVoxels, nClasses

@compute @workgroup_size(256)
fn voxelArgmax(@builtin(global_invocation_id) gid: vec3u) {
    let v = gid.x;
    if v >= params.x { return; }

    let base      = v * params.y;
    var bestClass = 0u;
    var bestCount = 0u;

    // Razred 0 je "neklasificirano" — ga preskočimo
    for (var c = 1u; c < params.y; c++) {
        let cnt = voxelClassVotes[base + c];
        if cnt > bestCount {
            bestCount = cnt;
            bestClass = c;
        }
    }

    voxelWinner[v] = bestClass;
}
