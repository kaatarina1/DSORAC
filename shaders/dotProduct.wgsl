@group(0) @binding(0) var textureA: texture_2d<f32>;
@group(0) @binding(1) var textureB: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outputBuffer : array<f32>;

// Shared memory for the workgroup reduction
var<workgroup> localSums: array<vec4<f32>, 64>; // For 8x8 workgroup

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let size = textureDimensions(textureA);
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return; 
    }

    let coords = vec2<i32>(global_id.xy);

    let color1 = textureLoad(textureA, coords, 0);
    let color2 = textureLoad(textureB, coords, 0);

    // Compute channel-wise products
    let partialDotR = color1.r * color2.r;
    let partialDotG = color1.g * color2.g;
    let partialDotB = color1.b * color2.b;
    let partialDotA = color1.a * color2.a;

    // Write the partial results to the output buffer
    let index = (global_id.y * size.x + global_id.x) * 4;
    outputBuffer[index + 0] = partialDotR;
    outputBuffer[index + 1] = partialDotG;
    outputBuffer[index + 2] = partialDotB;
    outputBuffer[index + 3] = partialDotA;
}