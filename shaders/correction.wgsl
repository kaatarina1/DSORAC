@group(0) @binding(0) var reconstructionTexture: texture_2d<f32>;
@group(0) @binding(1) var correctionTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<bgra8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = textureDimensions(outputTexture);
    
    // Check if within texture bounds
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }

    // Calculate texture coordinates
    let outCoord = vec2<i32>(global_id.xy);
    
    // Sample reconstruction and correction textures
    let reconstruction = textureLoad(reconstructionTexture, outCoord, 0);
    let correction = textureLoad(correctionTexture, outCoord, 0);
    
    // Add correction to reconstruction
    let result = reconstruction + correction;
    
    textureStore(outputTexture, outCoord, result);
}