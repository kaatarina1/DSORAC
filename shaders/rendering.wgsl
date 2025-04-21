struct Point {
    position: vec3f,
    color: u32,
}

@group(0) @binding(0) var<storage,read> points: array<Point>;
@group(1) @binding(0) var<uniform> matrix: mat4x4f;
@group(2) @binding(0) var<uniform> depthRange: vec2f; 

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) @interpolate(flat) color: vec4f,
}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput {
    let point = &points[index];

    var output: VertexOutput;
    // output.clipPosition = matrix * vec4f((*point).position, 1);
    let depth = (matrix * vec4f((*point).position, 1)).z; // Extract depth value

    // Discard points outside the selected depth range
    if (depth < depthRange.x || depth > depthRange.y) {
        output.clipPosition = vec4f(0.0, 0.0, 2.0, 0.0); // Move outside clip space
    } else {
        output.clipPosition = matrix * vec4f((*point).position, 1);
    }
    output.color = unpack4x8unorm((*point).color);
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}
