@group(0) @binding(0) var maskTex : texture_2d<u32>;
@group(0) @binding(1) var coordOut : texture_storage_2d<rgba32sint, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(maskTex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let m = textureLoad(maskTex, vec2<i32>(gid.xy), 0).x;
  if (m != 0u) {
    textureStore(coordOut, vec2<i32>(gid.xy), vec4<i32>(i32(gid.x), i32(gid.y), 0, 1));
  } else {
    textureStore(coordOut, vec2<i32>(gid.xy), vec4<i32>(-1, -1, 0, 1));
  }
}