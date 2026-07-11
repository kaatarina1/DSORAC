struct Point {
    position:      vec3f,
    color:         u32,
    normal:        vec3f,
    depth:         f32,
    classColor:    u32,
    _pad0:         u32,
    lasClassColor: u32,
    _pad2:         u32,
}

struct SceneParams {
    targetPosition: vec4f,
    cameraPos:      vec4f,
    pointSize:      f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<uniform> useClassColors: u32;
@group(1) @binding(0) var<uniform> matrix:     mat4x4f;
@group(1) @binding(1) var<uniform> viewMatrix: mat4x4f;
@group(2) @binding(0) var<uniform> depthRange: vec2f;
@group(3) @binding(0) var<uniform> scene:      SceneParams;

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) @interpolate(flat) color: vec4f,
    @location(1) uv: vec2f,
    @location(2) @interpolate(flat) isTarget: f32,
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
    let corner = vid % 6u;
    let point = &points[pid];
    var output: VertexOutput;

    let toCamera = scene.cameraPos.xyz - (*point).position;
    let toCameraDir = normalize(toCamera);

    // Guard against zero/invalid normals — fall back to camera-facing
    let rawNormal = (*point).normal;
    let normalLen = length(rawNormal);
    var faceNormal: vec3f;
    if (normalLen < 0.001) {
        faceNormal = toCameraDir;          // treat as billboard
    } else {
        faceNormal = rawNormal / normalLen; // safe normalize
        // Flip if back-facing
        if (dot(faceNormal, toCamera) < 0.0) {
            faceNormal = -faceNormal;
        }
    }

    var worldUp = vec3f(0.0, 1.0, 0.0);
    if (abs(dot(faceNormal, worldUp)) > 0.99) {
        worldUp = vec3f(0.0, 0.0, 1.0);
    }
    let billRight = normalize(cross(worldUp, faceNormal));
    let billUp = normalize(cross(faceNormal, billRight));

    let uv = CORNERS[corner];
    let worldPos = (*point).position
                 + billRight * uv.x * scene.pointSize
                 + billUp    * uv.y * scene.pointSize;

    let viewPos = viewMatrix * vec4f((*point).position, 1.0);
    let linearDepth = -viewPos.z;

    if (linearDepth < depthRange.x || linearDepth > depthRange.y) {
        output.clipPosition = vec4f(0.0, 0.0, 2.0, 0.0);
    } else {
        output.clipPosition = matrix * vec4f(worldPos, 1.0);
    }

    output.uv = uv;
    output.isTarget = select(0.0, 1.0, distance((*point).position, scene.targetPosition.xyz) < 0.02);
    var pickedColor: u32;
    if (useClassColors == 1u) {
        pickedColor = (*point).classColor;
    } else if (useClassColors == 2u) {
        pickedColor = (*point).lasClassColor;
    } else {
        pickedColor = (*point).color;
    }
    output.color = unpack4x8unorm(pickedColor);
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
    let r = dot(input.uv, input.uv);
    if (r > 1.0) { discard; }
    return vec4f(input.color.rgb, input.color.a);
}
