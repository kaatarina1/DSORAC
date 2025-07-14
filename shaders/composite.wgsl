@group(0) @binding(0) var reconstructionTextures: texture_2d_array<f32>;
@group(0) @binding(1) var sdfTextures: texture_2d_array<f32>;
@group(0) @binding(2) var pointsTextures: texture_2d_array<f32>;
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba32float, write>;

@group(1) @binding(0) var<uniform> uniforms: vec3<u32>;
@group(1) @binding(1) var<storage> depths: array<f32>;
@group(1) @binding(2) var<storage, read_write> iDepths: array<u32>;
@group(1) @binding(3) var<storage, read_write> iColorX: array<u32>;
@group(1) @binding(4) var<storage, read_write> iColorY: array<u32>;

fn standard_normal_cdf(z: f32) -> f32 {
    // Handle symmetry: Φ(-z) = 1 - Φ(z)
    let sign = select(1.0, -1.0, z < 0.0);
    let abs_z = abs(z);

    // Constants for Abramowitz-Stegun approximation
    const a1: f32 = 0.0498673470;
    const a2: f32 = 0.0211410061;
    const a3: f32 = 0.0032776263;
    const a4: f32 = 0.0000380036;
    const a5: f32 = 0.0000488906;
    const a6: f32 = 0.0000053830;

    let t = 1.0 + abs_z * (a1 + abs_z * (a2 + abs_z * (a3 + abs_z * (a4 + abs_z * (a5 + abs_z * a6)))));
    let t_pow_16 = pow(t, -16.0);
    let phi = 1.0 - 0.5 * t_pow_16;

    return select(phi, 1.0 - phi, z < 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size1 = textureDimensions(reconstructionTextures);
    let size2 = textureDimensions(sdfTextures);
    let size3 = textureDimensions(pointsTextures);
    iDepths[0] = 0;
    iColorX[0] = 0;
    iColorY[0] = 0;

    let index = global_id.xy;
    let temp = uniforms;
    let size = textureDimensions(outputTexture);
    if (index.x >= size.x || index.y >= size.y) {
        return;
    }

    // var indexD: array<u32>;

    // var indexC = array<vec2u, 2>(
    //     vec2u(0u, 0u), vec2u(0u, 0u)
    // );
    var indexD = array<u32, 30>(
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
            0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u
        );
    var indexC = array<vec2u, 30>(
        vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u),
        vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u),
        vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u),
        vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u),
        vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u),
        vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u), vec2u(0u, 0u)
    );

    let numberOfLayers = textureNumLayers(reconstructionTextures);

    var count = 0;
    var meanSum = 0f;
    var sigmaSum = 0f;

    for (var i = 0u; i < numberOfLayers; i++) {
        let sdf = textureLoad(sdfTextures, index, i, 0);
        let depth = depths[i];
        let w = 1e-30 + exp(-(sdf.a * sdf.a));
        let distance = 1 / w;

        indexD[count] = i;
        indexC[count] = index;
        // atomicStore(&iDepths[count], i);
        // atomicStore(&iColorX[count], index.x);
        // atomicStore(&iColorY[count], index.y);
        count += 1;
        meanSum += depth;
        sigmaSum += sdf.a * sdf.a;
    }

    let sqrtSigma = sqrt(sigmaSum);

    var color = vec4f(1.0, 1.0, 1.0, 1.0);
    for (var i = 0u; i < u32(count); i++) {
        let indexDepth = indexD[i];
        let indexColor = indexC[i];
        // let indexDepth = atomicLoad(&iDepths[i]);
        // let indexColorX = atomicLoad(&iColorX[i]);
        // let indexColorY = atomicLoad(&iColorY[i]);
        // let coord = vec2u(indexColorX, indexColorY);
        let depth = depths[indexDepth];
        let mean = depth - (meanSum - depth);
        let zScore = (0 - mean) / sqrtSigma;

        let p = standard_normal_cdf(zScore);

        var col = textureLoad(reconstructionTextures, indexColor, indexDepth, 0);
        let sdf = textureLoad(sdfTextures, indexColor, indexDepth, 0);

        let alpha = exp(-(700 * sdf.a * sdf.a));

        col.a *= alpha;
        col = vec4f(col.rgb * col.a, col.a);
        color = p * col + (1 - p * col.a) * color;
        // color = col;
        color = vec4f(color.rgb * color.a, color.a);
    }

    textureStore(outputTexture, index, color * 255.0);
}
            
        