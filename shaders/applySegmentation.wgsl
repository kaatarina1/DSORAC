// Prekopiramo zmagovalni razredi iz winner buffer-jem, 
// ki ima razrede zapisane na orginalnih indexih točk (pred sortiranjem)
//_pad0 za vsako točko hrani njen orginalni index pred sortiranjem


struct Point {
    position:   vec3f,
    color:      u32,
    normal:     vec3f,
    depth:      f32,
    classColor: u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}

@group(0) @binding(0) var<storage, read_write> points:  array<Point>;
@group(0) @binding(1) var<storage, read>       winner:  array<u32>;
@group(0) @binding(2) var<uniform>             nPoints: u32;

@compute @workgroup_size(256)
fn apply(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if i >= nPoints { return; }
    points[i].classColor = winner[points[i]._pad0];
}
