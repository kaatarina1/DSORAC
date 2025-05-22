@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> boundary: u32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputSize = textureDimensions(outputTexture);
    let inputSize = textureDimensions(inputTexture);

    // Exit if out of bounds
    if (global_id.x >= outputSize.x || global_id.y >= outputSize.y) {
        return;
    }

    // Calculate the base coordinate in the input texture (2x2 block per output pixel)
    let inputBaseCoord = vec2<u32>(global_id.xy) * 2u;

    // Sample 4 input pixels (bilinear interpolation would also work)
    let color0 = textureLoad(inputTexture, vec2<i32>(inputBaseCoord), 0);
    let color1 = textureLoad(inputTexture, vec2<i32>(inputBaseCoord + vec2<u32>(1u, 0u)), 0);
    let color2 = textureLoad(inputTexture, vec2<i32>(inputBaseCoord + vec2<u32>(0u, 1u)), 0);
    let color3 = textureLoad(inputTexture, vec2<i32>(inputBaseCoord + vec2<u32>(1u, 1u)), 0);

    // Average the 4 samples
    let averagedColor = (color0 + color1 + color2 + color3) * 0.25;

    // Apply boundary condition (optional)
    let outputColor = select(
        averagedColor,
        vec4<f32>(0.0, 0.0, 0.0, averagedColor.a),
        boundary != 0u
    );

    // Write to output
    textureStore(outputTexture, vec2<i32>(global_id.xy), outputColor);
}