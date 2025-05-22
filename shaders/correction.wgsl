@group(0) @binding(0) var reconstructionTexture: texture_2d<f32>;
@group(0) @binding(1) var correctionTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outputSize = textureDimensions(outputTexture);
    let correctionSize = textureDimensions(correctionTexture);

    if (global_id.x >= outputSize.x || global_id.y >= outputSize.y) {
        return;
    }

    let outCoord = vec2<i32>(global_id.xy);
    let reconstruction = textureLoad(reconstructionTexture, outCoord, 0);

    // Scale correction texture coords to match its smaller size
    let correctionCoord = vec2<i32>(global_id.xy / 2u);
    let correction = textureLoad(correctionTexture, correctionCoord, 0);

    textureStore(outputTexture, outCoord, reconstruction + correction);
}