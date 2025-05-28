@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var textureSampler: sampler;
@group(0) @binding(3) var<uniform> boundary: u32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputSize = textureDimensions(outputTexture);
    let inputSize = textureDimensions(inputTexture);

    if (global_id.x >= outputSize.x|| global_id.y >= outputSize.y) {
        return;
    }

    let coord = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(outputSize);

    let color = textureSampleLevel(inputTexture, textureSampler, coord, 0.0);
    var outputColor = vec4f(0);
    if (boundary == 1u) {
        outputColor = vec4f(0.0, 0.0, 0.0, color.a);
    } else {
        outputColor = color;
    }
    textureStore(outputTexture, vec2<i32>(global_id.xy), outputColor);
}