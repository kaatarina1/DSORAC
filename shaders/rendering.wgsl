struct Point {
    position: vec3f,
    color: u32,
}

@group(0) @binding(0) var<storage,read> points: array<Point>;
@group(1) @binding(0) var<uniform> matrix: mat4x4f;
@group(2) @binding(0) var<uniform> depthRange: vec2f; 
@group(3) @binding(0) var<uniform> targetPosition: vec3f; // Add this line

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) @interpolate(flat) color: vec4f,
    @location(1) @interpolate(flat) isTarget: f32, // Add this
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

    // output.clipPosition = matrix * vec4f((*point).position, 1);

    let isTarget = distance((*point).position, targetPosition) < 0.02; // Check if point is near target
    output.isTarget = select(0.0, 1.0, isTarget); // Set to 1.0 if near target, else 0.0
    output.color = unpack4x8unorm((*point).color);
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
    //  if (input.isTarget > 0.5) {
    //     return vec4f(1.0, 0.0, 0.0, 1.0); // Bright red
    // }
    return input.color.rgba;
}
