struct Point {
    position: vec3f,
    color: u32,
    normal: vec3f,
    depth: f32,
}

@group(0) @binding(0) var<storage, read_write> points: array<Point>;
@group(1) @binding(0) var<uniform> projectViewMatrix: mat4x4f;
@group(1) @binding(1) var<uniform> numberOfPoints: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= numberOfPoints) { 
        points[i].depth = -1e30;
        return;
    }

    let viewPos = projectViewMatrix * vec4f(points[i].position, 1.0);
    let depth = viewPos.z / viewPos.w;
    points[i].depth = depth;
}