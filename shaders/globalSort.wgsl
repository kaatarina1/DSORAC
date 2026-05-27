struct Point {
    position:   vec3f,
    color:      u32,
    normal:     vec3f,
    depth:      f32,
    classColor: u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
}


@group(0) @binding(0) var<storage, read_write> points: array<Point>;
@group(0) @binding(1) var<uniform> numberOfPoints : u32;
@group(0) @binding(2) var<uniform> mergeParams : vec2u; // k, j

@compute
@workgroup_size(256)
fn mergePass(@builtin(global_invocation_id) globalId: vec3u) {
    let globalIndex = globalId.x;

    let k = mergeParams.x;
    let j = mergeParams.y;

    let ixj = globalIndex ^ j;

    if (ixj > globalIndex && ixj < numberOfPoints)
    {
        let ascending = (globalIndex & k) == 0;
        let currDepth = points[globalIndex].depth;
        let otherDepth = points[ixj].depth;

        let swap = (ascending && currDepth < otherDepth) || (!ascending && currDepth > otherDepth);

        if (swap)
        {
            let temp = points[globalIndex];
            points[globalIndex] = points[ixj];
            points[ixj] = temp;
        }
    }
}
