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

@group(0) @binding(0) var<storage, read_write> points: array<Point>;
@group(1) @binding(0) var<uniform> projectViewMatrix: mat4x4f;
@group(1) @binding(1) var<uniform> numberOfPoints: u32;
@group(1) @binding(2) var<uniform> isSpherical: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= numberOfPoints) {
        points[i].depth = -1e30;
        return;
    }

    let viewPos = projectViewMatrix * vec4f(points[i].position, 1.0);
    if (isSpherical == 1u) {
        // Za sferične (panoramske) zajeme ni prave projekcijske matrike -
        // projektViewMatrix je tu dejanska view matrika, zato sortiramo
        // po radialni razdalji od kamere (enako kot v rendering_gaussians.wgsl).
        points[i].depth = length(viewPos.xyz);
    } else {
        points[i].depth = viewPos.z / viewPos.w;
    }
}