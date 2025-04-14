@group(0) @binding(0) var uCapture: texture_2d<f32>;
@group(0) @binding(1) var uReconstruction: texture_2d<f32>;
@group(0) @binding(2) var oReconstruction: texture_storage_2d<bgra8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texSize = vec2<f32>(textureDimensions(uCapture));
    let texelSize = vec2<f32>(1.0 / texSize.x, 1.0 / texSize.y);
    let vPosition = (vec2<f32>(global_id.xy) + 0.5) / texSize;

    // Bounds check
    if (global_id.x >= u32(texSize.x) || global_id.y >= u32(texSize.y)) {
        return;
    }

    // Sample points texture
    let points = textureLoad(uCapture, vec2<i32>(global_id.xy), 0);
    if (points.a > 0.0) {
        textureStore(oReconstruction, vec2<i32>(global_id.xy), points);
        return;
    }

    // Neighbor sampling
    let c = textureLoad(uReconstruction, vec2<i32>(global_id.xy), 0);
    let f = 2.0;
    let l = textureLoad(uReconstruction, vec2<i32>(global_id.xy) + vec2<i32>(-1, 0), 0);
    let r = textureLoad(uReconstruction, vec2<i32>(global_id.xy) + vec2<i32>(1, 0), 0);
    let d = textureLoad(uReconstruction, vec2<i32>(global_id.xy) + vec2<i32>(0, -1), 0);
    let u = textureLoad(uReconstruction, vec2<i32>(global_id.xy) + vec2<i32>(0, 1), 0);

    let h = 1.0 / max(texSize.x - 1.0, texSize.y - 1.0);
    let average = 0.25 * (l + r + d + u - h * h * f);


    // Write result to the output texture
    textureStore(oReconstruction, vec2<i32>(global_id.xy), average);
}
