struct ParamsF { normFactor : f32, width : f32, height : f32, pad : f32 };
@group(0) @binding(2) var<uniform> params : ParamsF;

@group(0) @binding(0) var coordIn : texture_2d<i32>;
@group(0) @binding(1) var sdfOut : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= u32(params.width) || gid.y >= u32(params.height)) { return; }
  let seed = textureLoad(coordIn, vec2<i32>(gid.xy), 0).xy;
  var dist = 0.0;
  if (seed.x >= 0) {
    let dx = f32(seed.x) - f32(gid.x);
    let dy = f32(seed.y) - f32(gid.y);
    dist = sqrt(dx*dx + dy*dy);
  } else {
    dist = sqrt(params.width*params.width + params.height*params.height);
  }
  let a = dist * params.normFactor;
  textureStore(sdfOut, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, a));
}