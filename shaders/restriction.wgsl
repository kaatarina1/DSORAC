@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> boundary: u32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = textureDimensions(outputTexture);
    
    // Check if within texture bounds
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }

    // Calculate texture coordinates
    let outCoord = vec2<i32>(global_id.xy);
    let texCoord = vec2<f32>(f32(global_id.x) / f32(size.x), f32(global_id.y) / f32(size.y));
    
    // Sample input texture
    let color = textureLoad(inputTexture, outCoord, 0);
    
    // Apply boundary conditions if needed
    if (boundary != 0u) {
        // Boundary conditions on coarser grids are homogeneous
        textureStore(outputTexture, outCoord, vec4<f32>(0.0, 0.0, 0.0, color.a));
    } else {
        textureStore(outputTexture, outCoord, color);
    }
}