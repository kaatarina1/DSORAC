// Za vsak piksel preberemo vidno točko na tem pikslu
// ter ID razreda iz mask bufferja in dodamo en glas temu razredu

@group(0) @binding(0) var<storage, read_write> votes: array<atomic<u32>>;
@group(0) @binding(1) var indexTex: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4u;  // nClasses, width, height, _pad

@compute @workgroup_size(16, 16)
fn vote(@builtin(global_invocation_id) gid: vec3u) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.y || y >= params.z { return; }

    // 0 = ni izrisane (vidne) nobene točke na tem mestu
    let pointIdx = textureLoad(indexTex, vec2u(x, y), 0u).r;
    if pointIdx == 0u { return; }

    let classId = mask[y * params.y + x];
    if classId == 0u || classId >= params.x { return; }

    atomicAdd(&votes[(pointIdx - 1u) * params.x + classId], 1u);
}
