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
    isOrtho: f32, // isOrtho
    isSpherical: f32, // isSpherical
    far: f32, // far
}

const PI: f32 = 3.14159265359;

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

    // če je cameraPos.w != 0 to pomeni da uporabljamo ortografsko kamero,
    // v tem primeru je pozicija kamere konstantna, se pravi vsi diski morejo 
    // imeti enako orientacijo -> navzgor
    let toCamera = select(
        normalize(scene.cameraPos.xyz - (*point).position),
        scene.cameraPos.xyz,
        scene.isOrtho > 0.5
    );

    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(dot(toCamera, up)) > 0.99) {
        up = vec3f(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(up, toCamera));
    let tangent = normalize(cross(toCamera, right));

    let uv = CORNERS[corner];
    let worldPos = (*point).position
                 + right   * uv.x * scene.pointSize
                 + tangent * uv.y * scene.pointSize;

    let viewPos = viewMatrix * vec4f((*point).position, 1.0);
    let linearDepth = -viewPos.z;

    if (scene.isSpherical > 0.5) {
        let viewPosOffset = viewMatrix * vec4f(worldPos, 1.0);
        let distCenter = length(viewPos.xyz);
        let dist = length(viewPosOffset.xyz);
        if (dist > scene.far) {
            output.clipPosition = vec4f(0.0, 0.0, 2.0, 0.0);
        } else {
            let dirCenter = viewPos.xyz / max(distCenter, 1e-6); // Normalize 
            let lonCenter = atan2(dirCenter.x, -dirCenter.z);
            let dir = viewPosOffset.xyz / max(dist, 1e-6);
            var lon = atan2(dir.x, -dir.z);
            let diff = lon - lonCenter;
            let wrapped = diff - 2.0 * PI * round(diff / (2.0 * PI));
            lon = lonCenter + wrapped;
            let lat = asin(clamp(dir.y, -1.0, 1.0));
            output.clipPosition = vec4f(lon / PI, lat / (PI * 0.5), clamp(dist / max(scene.far, 1e-6), 0.0, 1.0), 1.0);
        }
    } else {
        // Discard points outside the selected linear depth range
        if (linearDepth < depthRange.x || linearDepth > depthRange.y) {
            output.clipPosition = vec4f(0.0, 0.0, 2.0, 0.0); // Move outside clip space
        } else {
            output.clipPosition = matrix * vec4f(worldPos, 1.0);
        }
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
    if (dot(input.uv, input.uv) > 1.0) {
        discard;
    }
    return input.color;
}
