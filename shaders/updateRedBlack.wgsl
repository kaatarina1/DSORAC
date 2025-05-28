@group(0) @binding(0) var uCapture: texture_2d<f32>;
@group(0) @binding(1) var uReconstruction: texture_2d<f32>;
@group(0) @binding(2) var oReconstruction: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var f: texture_2d<f32>;
@group(0) @binding(4) var textureSampler: sampler;
@group(0) @binding(5) var<uniform> uRedBlack: u32;
@group(0) @binding(6) var<uniform> uOmega: f32;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = textureDimensions(uCapture);
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }
    
    let texSize = vec2<f32>(size);
    let texelSize = 1.0 / texSize;
    let vPosition = (vec2<f32>(global_id.xy) + 0.5) / texSize;

    
    let points = textureSampleLevel(uCapture, textureSampler, vPosition, 0.0);
    if (points.a > 0.0) {
        textureStore(oReconstruction, vec2<i32>(global_id.xy), points);
        return;
    }

    let c = textureSampleLevel(uReconstruction, textureSampler, vPosition, 0.0);
    
    let isRed = u32(vPosition.x * texSize.x + vPosition.y * texSize.y) % 2u == 0u;
    
    if ((uRedBlack == 0u && !isRed) || (uRedBlack == 1u && isRed)) {
        textureStore(oReconstruction, vec2<i32>(global_id.xy), c);
        return;
    }

    let h = 1.0 / f32(max(size.x, size.y) - 1);
    
    let f = textureSampleLevel(f, textureSampler, vPosition, 0.0);    
    let l = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2(-1.0, 0.0) * texelSize, 0.0);
    let r = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2(1.0, 0.0) * texelSize, 0.0);
    let d = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2(0.0, -1.0) * texelSize, 0.0);
    let u = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2(0.0, 1.0) * texelSize, 0.0);

    // SOR iteration formula
    let average = 0.25 * (l + r + d + u - h*h * f);
    let result = uOmega * average + (1.0 - uOmega) * c;
    
    textureStore(oReconstruction, vec2<i32>(global_id.xy), result);
}