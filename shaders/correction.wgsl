@group(0) @binding(0) var reconstructionTexture: texture_2d<f32>;
@group(0) @binding(1) var correctionTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var textureSampler: sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputSize = textureDimensions(outputTexture);
    let correctionSize = textureDimensions(correctionTexture);

    if (global_id.x >= outputSize.x || global_id.y >= outputSize.y) {
        return;
    }

    let outCoord = (vec2<f32>(global_id.xy) + 0.5) / vec2<f32>(outputSize);
    let reconstruction = textureSampleLevel(reconstructionTexture, textureSampler, outCoord, 0.0);
    let correction = textureSampleLevel(correctionTexture, textureSampler, outCoord, 0.0);

    textureStore(outputTexture, vec2<i32>(global_id.xy), reconstruction + correction);
}