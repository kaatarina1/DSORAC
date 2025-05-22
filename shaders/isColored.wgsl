@group(0) @binding(0) var uInput: texture_2d<f32>;
@group(0) @binding(1) var oOutput: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = textureDimensions(uInput);
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }
    let c = textureLoad(uInput, vec2<i32>(global_id.xy), 0);
    var value = 0.0;
    if (c.a > 0.0) {
        value = 1.0;
    }
    textureStore(oOutput, vec2<i32>(global_id.xy), vec4<u32>(u32(value), 0, 0, 0));
}