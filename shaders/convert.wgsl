@group(0) @binding(0) var uInput: texture_2d<f32>;
@group(0) @binding(1) var oOutput: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let value = textureLoad(uInput, vec2<i32>(global_id.xy), 0);
    textureStore(oOutput, vec2<i32>(global_id.xy), vec4<f32>(value));
}