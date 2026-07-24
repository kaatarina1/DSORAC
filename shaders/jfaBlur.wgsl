struct BlureParams {
  sigma: f32,
  radius: f32,
  axisX: f32,
  axisY: f32,
}

@group(0) @binding(0) var sdfIn : texture_2d<f32>;
@group(0) @binding(1) var sdfOut : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: BlureParams;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(sdfIn);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let radius = i32(params.radius);
  let sigma2 = max(params.sigma * params.sigma, 1e-4);
  let dir = vec2<i32>(i32(params.axisX), i32(params.axisY));

  var sum = 0.0;
  var wsum = 0.0;
  for (var i = -radius; i <= radius; i++) {
    let nx = clamp(i32(gid.x) + dir.x * i, 0, i32(dims.x) - 1);
    let ny = clamp(i32(gid.y) + dir.y * i, 0, i32(dims.y) - 1);
    let w = exp(-f32(i * i) / (2.0 * sigma2));
    sum += textureLoad(sdfIn, vec2<i32>(nx, ny), 0).a * w;
    wsum += w;
  }
  let blurred = sum / wsum;
  textureStore(sdfOut, vec2<i32>(gid.xy), vec4<f32>(1.0, 1.0, 1.0, blurred));
}