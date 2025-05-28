@group(0) @binding(0) var uCapture: texture_2d<f32>;
@group(0) @binding(1) var uReconstruction: texture_2d<f32>;
@group(0) @binding(2) var textureSampler: sampler;
@group(0) @binding(3) var oReconstruction: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texSize = vec2<f32>(textureDimensions(uCapture));
    let texelSize = vec2<f32>(1.0 / texSize.x, 1.0 / texSize.y);
    let vPosition = (vec2<f32>(global_id.xy) + 0.5) / texSize;

    if (global_id.x >= u32(texSize.x) || global_id.y >= u32(texSize.y)) {
        return;
    }

    let points = textureSampleLevel(uCapture, textureSampler, vPosition, 0.0);
    if (points.a > 0.0) {
        textureStore(oReconstruction, vec2<i32>(global_id.xy), points);
        return;
    }

    let c = textureSampleLevel(uReconstruction, textureSampler, vPosition, 0);
    let l = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2<f32>(-1.0, 0.0) * texelSize, 0.0);
    let r = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2<f32>(1.0, 0.0) * texelSize, 0.0);
    let d = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2<f32>(0.0, -1.0) * texelSize, 0.0);
    let u = textureSampleLevel(uReconstruction, textureSampler, vPosition + vec2<f32>(0.0, 1.0) * texelSize, 0.0);

    let average = 0.25 * (l + r + d + u);


    textureStore(oReconstruction, vec2<i32>(global_id.xy), average);
}
