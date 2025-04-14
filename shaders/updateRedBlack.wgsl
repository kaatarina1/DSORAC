@group(0) @binding(0) var uCapture: texture_2d<f32>;
@group(0) @binding(1) var uReconstruction: texture_2d<f32>;
@group(0) @binding(2) var oReconstruction: texture_storage_2d<bgra8unorm, write>;
@group(0) @binding(3) var<uniform> uRedBlack: u32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let omega = 1.9;  // Should match your WebGL's omega value
    
    // Early exit if outside texture dimensions
    let texSize = vec2<u32>(textureDimensions(uCapture));
    if (global_id.x >= texSize.x || global_id.y >= texSize.y) {
        return;
    }
    
    // Sample points texture
    let points = textureLoad(uCapture, vec2<i32>(global_id.xy), 0);
    if (points.a > 0.0) {
        textureStore(oReconstruction, vec2<i32>(global_id.xy), points);
        return;
    }

    // Current value
    let c = textureLoad(uReconstruction, vec2<i32>(global_id.xy), 0);
    
    // Simplified checkerboard pattern - matches WebGL implementation
    let total = f32(global_id.x) + f32(global_id.y);
    let isRed = u32(total) % 2u == 0u;
    
    // This matches the WebGL's "isRed ^^ uRedBlack" logic
    if ((uRedBlack == 1u && isRed) || (uRedBlack == 0u && !isRed)) {
        textureStore(oReconstruction, vec2<i32>(global_id.xy), c);
        return;
    }

    let h = 1.0 / f32(max(texSize.x, texSize.y) - 1u);
    
    // Define a constant for the f value (matching WebGL)
    let f = vec4<f32>(2.0, 2.0, 2.0, 2.0);
    
    // Compute neighbor positions with safe boundary handling
    let lPos = vec2<i32>(max(i32(global_id.x) - 1, 0), i32(global_id.y));
    let rPos = vec2<i32>(min(i32(global_id.x) + 1, i32(texSize.x) - 1), i32(global_id.y));
    let dPos = vec2<i32>(i32(global_id.x), max(i32(global_id.y) - 1, 0));
    let uPos = vec2<i32>(i32(global_id.x), min(i32(global_id.y) + 1, i32(texSize.y) - 1));
    
    let l = textureLoad(uReconstruction, lPos, 0);
    let r = textureLoad(uReconstruction, rPos, 0);
    let d = textureLoad(uReconstruction, dPos, 0);
    let u = textureLoad(uReconstruction, uPos, 0);

    // SOR iteration formula
    let average = 0.25 * (l + r + d + u - h*h * f);
    let result = omega * average + (1.0 - omega) * c;
    
    textureStore(oReconstruction, vec2<i32>(global_id.xy), result);
}