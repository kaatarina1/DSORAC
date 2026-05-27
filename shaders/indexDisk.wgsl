struct Point {
    position:   vec3f,
    color:      u32,
    normal:     vec3f,
    depth:      f32,
    classColor: u32,
    _pad0:      u32,   // orginalen indeks točke
    _pad1:      u32,
    _pad2:      u32,
}

struct IndexScene {
    mvp:    mat4x4f,
    imgW:   f32,   // višina render targeta v pikslih
    imgH:   f32,   // širina render targeta v pikslih
    radius: f32,   // radij diksa v pikslih
    _p0:    f32,
}

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(1) @binding(0) var<uniform> scene: IndexScene;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) idx: u32,
    @location(1) uv: vec2f,
}

const CORNERS = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0, -1.0),
    vec2f( 1.0,  1.0),
    vec2f(-1.0,  1.0),
);

@vertex
fn vertex(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let pid = vid / 6u;
    let point = &points[pid];
    let uv  = CORNERS[vid % 6u];
    var output: VertexOutput;

    // Projeciramo center na clip space.
    let centerClip = scene.mvp * vec4f((*point).position, 1.);

    // Določimo, kje vse na sliki bi bil viden disk
    // NDC offset = radij / (0.5 širine/višine v pikslih)
    // Pomnožimo s clip.w, da je po perspektivni delitvi odmik v NDC natančen.
    // Globina (clip.z / clip.w) se ohrani, saj vsak piksel diska ima ENAKO globino kot
    // središče točke, zato test globine pravilno določi najbližjo točko
    // brez prekrivanja.
    let ndcDx = scene.radius / (scene.imgW * 0.5);
    let ndcDy = scene.radius / (scene.imgH * 0.5);
    output.pos = vec4f(
        centerClip.xy + uv * vec2f(ndcDx, ndcDy) * centerClip.w,
        centerClip.zw,
    );
    output.idx = (*point)._pad0;
    output.uv  = uv;
    return output;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) u32 {
    if dot(in.uv, in.uv) > 1. { discard; }
    return in.idx + 1u;   // 0 = neklasificirane, idx+1 = vidne točke
}
