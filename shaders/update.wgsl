@group(0) @binding(0) var imageA: texture_2d<f32>;
@group(0) @binding(1) var imageB: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> rgba: vec4f;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = textureDimensions(outputTexture);

    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }

    let coord = vec2<i32>(global_id.xy);

    let a = textureLoad(imageA, coord, 0);
    let b = textureLoad(imageB, coord, 0);

    let color = vec4f(a.r + rgba.r * b.r, a.g + rgba.g * b.g, a.b + rgba.b * b.b, a.a + rgba.a * b.a);
   
    textureStore(outputTexture, coord, color);
}