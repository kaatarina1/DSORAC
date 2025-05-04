@group(0) @binding(0) var uCapture: texture_2d<f32>;
@group(0) @binding(1) var uReconstruction: texture_2d<f32>;
@group(0) @binding(2) var oReconstruction: texture_storage_2d<bgra8unorm, write>;
@group(0) @binding(3) var<uniform> uRedBlack: u32;
@group(0) @binding(4) var<uniform> uOmega: f32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texSize = textureDimensions(uCapture);
    if (global_id.x >= texSize.x || global_id.y >= texSize.y) {
        return;
    }
    
    // Sample points texture
    let pos = vec2<i32>(global_id.xy);
    let points = textureLoad(uCapture, pos, 0);
    if (points.a > 0.0) {
        textureStore(oReconstruction, pos, points);
        return;
    }

    // Current value
    let c = textureLoad(uReconstruction, pos, 0);
    
    let isRed = (global_id.x + global_id.y) % 2u == 0u;
    
    // This matches the WebGL's "isRed ^^ uRedBlack" logic
    if ((uRedBlack == 0u && !isRed) || (uRedBlack == 1u && isRed)) {
        textureStore(oReconstruction, pos, c);
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
    let result = uOmega * average + (1.0 - uOmega) * c;
    
    textureStore(oReconstruction, pos, result);
}