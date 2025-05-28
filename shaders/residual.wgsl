@group(0) @binding(0) var pointsTexture: texture_2d<f32>;
@group(0) @binding(1) var reconstructionTexture: texture_2d<f32>;
@group(0) @binding(2) var f: texture_2d<f32>;
@group(0) @binding(3) var textureSampler: sampler;
@group(0) @binding(4) var outputTexture: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texSize = textureDimensions(outputTexture);
    let size = vec2<f32>(texSize.xy);
    let texelSize = 1.0 / size;
    let h = 1 / (size.x - 1.0);
    let vPosition = (vec2<f32>(global_id.xy) + 0.5) / size;
    
    if (global_id.x >= texSize.x - 1 || global_id.y >= texSize.y - 1) {
        return;
    }

    let c = textureSampleLevel(reconstructionTexture, textureSampler, vPosition, 0.0); 
    let f = textureSampleLevel(f, textureSampler, vPosition, 0.0); 
    let l = textureSampleLevel(reconstructionTexture, textureSampler, vPosition + vec2f(-1.0, 0.0) * texelSize, 0.0); 
    let r = textureSampleLevel(reconstructionTexture, textureSampler, vPosition + vec2f(1.0, 0.0) * texelSize, 0.0); 
    let d = textureSampleLevel(reconstructionTexture, textureSampler, vPosition + vec2f(0.0, -1.0) * texelSize, 0.0); 
    let u = textureSampleLevel(reconstructionTexture, textureSampler, vPosition + vec2f(0.0, 1.0) * texelSize, 0.0); 

    let points = textureSampleLevel(pointsTexture, textureSampler, vPosition, 0.0); 
    var outColor = vec4f(0);
    if (points.a < 1.0) {
        outColor = f - (l + r + d + u - 4.0 * c) / (h * h);
    } else {
        outColor = vec4f(0.0, 0.0, 0.0, 1.0);
    }

    textureStore(outputTexture, vec2<i32>(global_id.xy), outColor);
}
