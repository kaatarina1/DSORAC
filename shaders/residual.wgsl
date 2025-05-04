@group(0) @binding(0) var uOriginal: texture_2d<f32>;
@group(0) @binding(1) var uCurrent: texture_2d<f32>;
@group(0) @binding(2) var oResidual: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<storage, read_write> oResidualNorm: array<atomic<u32>>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texSize = textureDimensions(uOriginal);
    if (global_id.x >= texSize.x || global_id.y >= texSize.y) {
        return;
    }
    
    let pos = vec2<i32>(global_id.xy);
    let original = textureLoad(uOriginal, pos, 0);
    let current = textureLoad(uCurrent, pos, 0);

    if (original.a > 0.0) {
        return;
    }
    
    // Compute residual
    let residual = original - current;
    textureStore(oResidual, pos, residual);
    
    // Compute squared residual for norm calculation
    let residualSq = dot(residual.rgb, residual.rgb);
    let scaled = u32(residualSq * 1000.0); // Scale factor maintains some precision
    
    // Atomic add to residual norm
    atomicAdd(&oResidualNorm[0], scaled);
}