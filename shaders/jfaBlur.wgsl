@group(0) @binding(0) var sdfIn : texture_2d<f32>;
@group(0) @binding(1) var sdfOut : texture_storage_2d<rgba32float, write>;

const kernel : array<f32, 9> = array<f32, 9>(
  1.0, 2.0, 1.0,
  2.0, 4.0, 2.0,
  1.0, 2.0, 1.0
);

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(sdfIn);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  var sum = 0.0;
  var wsum = 0.0;
  var k = 0u;
  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      let nx = clamp(i32(gid.x) + ox, 0, i32(dims.x) - 1);
      let ny = clamp(i32(gid.y) + oy, 0, i32(dims.y) - 1);
      let w = kernel[k];
      sum += textureLoad(sdfIn, vec2<i32>(nx, ny), 0).a * w;
      wsum += w;
      k++;
    }
  }
  let blurred = sum / wsum;
  textureStore(sdfOut, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, blurred));
}