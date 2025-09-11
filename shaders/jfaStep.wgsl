struct Params { step : i32, width : u32, height : u32, pad : u32 };
@group(0) @binding(2) var<uniform> params : Params;

@group(0) @binding(0) var coordIn : texture_2d<i32>;
@group(0) @binding(1) var coordOut : texture_storage_2d<rgba32sint, write>;

fn sqr(x: f32) -> f32 { return x * x; }

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  var best = textureLoad(coordIn, vec2<i32>(gid.xy), 0).xy;
  var bestDist = 1e30;
  if (best.x >= 0) {
    let dx = f32(best.x - i32(gid.x));
    let dy = f32(best.y - i32(gid.y));
    bestDist = sqr(dx) + sqr(dy);
  }

  let s = params.step;
  for (var ox = -1; ox <= 1; ox++) {
    for (var oy = -1; oy <= 1; oy++) {
      if (ox == 0 && oy == 0) { continue; }
      let nx = i32(gid.x) + ox * s;
      let ny = i32(gid.y) + oy * s;
      if (nx < 0 || ny < 0 || u32(nx) >= params.width || u32(ny) >= params.height) {
        continue;
      }
      let seed = textureLoad(coordIn, vec2<i32>(nx, ny), 0).xy;
      if (seed.x < 0) { continue; }
      let dx = f32(seed.x - i32(gid.x));
      let dy = f32(seed.y - i32(gid.y));
      let d = sqr(dx) + sqr(dy);
      if (d < bestDist) { bestDist = d; best = seed; }
    }
  }

  textureStore(coordOut, vec2<i32>(gid.xy), vec4<i32>(best, 0, 1));
}