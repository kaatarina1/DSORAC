// Za vsako točko poišče top-3 razrede po številu glasov.
// topK[idx*3 + k] = (classId << 16) | min(count, 0xFFFF)
// Razred 0 (neklasificirano) je preskočen.

@group(0) @binding(0) var<storage, read>       votes:  array<u32>;
@group(0) @binding(1) var<storage, read_write> topK:   array<u32>;  // nPoints * 3
@group(0) @binding(2) var<uniform>             params: vec2u;        // nClasses, nPoints

@compute @workgroup_size(256)
fn topkArgmax(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if idx >= params.y { return; }

    let base = idx * params.x;

    var id0 = 0u; var cnt0 = 0u;
    var id1 = 0u; var cnt1 = 0u;
    var id2 = 0u; var cnt2 = 0u;

    // Razred 0 je "neklasificirano" — ga preskočimo
    for (var c = 1u; c < params.x; c++) {
        let cnt = votes[base + c];
        if cnt > cnt0 {
            id2 = id1; cnt2 = cnt1;
            id1 = id0; cnt1 = cnt0;
            id0 = c;   cnt0 = cnt;
        } else if cnt > cnt1 {
            id2 = id1; cnt2 = cnt1;
            id1 = c;   cnt1 = cnt;
        } else if cnt > cnt2 {
            id2 = c;   cnt2 = cnt;
        }
    }

    topK[idx * 3u + 0u] = (id0 << 16u) | min(cnt0, 0xFFFFu);
    topK[idx * 3u + 1u] = (id1 << 16u) | min(cnt1, 0xFFFFu);
    topK[idx * 3u + 2u] = (id2 << 16u) | min(cnt2, 0xFFFFu);
}
