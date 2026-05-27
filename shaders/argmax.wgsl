// Zgradi winner buffer, tako da vsako točko pogleda kateri razred ima največ glasov.
// winner buffer hrani razrede po orginalnem indexu točk (pred sortiranjem)

@group(0) @binding(0) var<storage, read>       votes:   array<u32>;
@group(0) @binding(1) var<storage, read>       palette: array<u32>;
@group(0) @binding(2) var<storage, read_write> winner:  array<u32>;
@group(0) @binding(3) var<uniform>             params:  vec2u;  // nClasses, nPoints

@compute @workgroup_size(256)
fn argmax(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if idx >= params.y { return; }

    let base      = idx * params.x;
    var bestClass = 0u;
    var bestCount = 0u;

    // Začne z razredom 1 — razred 0 je za neklasificirane točke
    for (var c = 1u; c < params.x; c++) {
        let cnt = votes[base + c];
        if cnt > bestCount {
            bestCount = cnt;
            bestClass = c;
        }
    }

    // palette[0] = siva (neklasificirano)
    // to se zgodi, ko je bestCount == 0
    winner[idx] = palette[bestClass];
}
