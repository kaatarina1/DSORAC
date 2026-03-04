struct Point {
    position: vec3f,
    color: u32,
    normal: vec3f,
    depth: f32
}

@group(0) @binding(0) var<storage,read> points: array<Point>;
@group(1) @binding(0) var<uniform> matrix: mat4x4f;
@group(1) @binding(1) var<uniform> viewMatrix: mat4x4f;
@group(2) @binding(0) var<uniform> depthRange: vec2f; 
@group(3) @binding(0) var<uniform> targetPosition: vec4f;

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) @interpolate(flat) color: vec4f,
    @location(1) @interpolate(flat) isTarget: f32,
}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
    let point = &points[index];

    var output: VertexOutput;

    // Compute linear view-space depth for filtering
    let viewPos = viewMatrix * vec4f((*point).position, 1);
    let linearDepth = -viewPos.z; // negate: camera looks along -Z

    // Discard points outside the selected linear depth range
    if (linearDepth < depthRange.x || linearDepth > depthRange.y) {
        output.clipPosition = vec4f(0.0, 0.0, 2.0, 0.0); // Move outside clip space
    } else {
        output.clipPosition = matrix * vec4f((*point).position, 1);
    }

    let isTarget = distance((*point).position, targetPosition.xyz) < 0.02;
    output.isTarget = select(0.0, 1.0, isTarget);
    output.color = unpack4x8unorm((*point).color);
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
    return input.color.rgba;
}
