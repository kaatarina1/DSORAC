@group(0) @binding(0) var reconstructionTextures: texture_2d_array<f32>;
@group(0) @binding(1) var sdfTextures: texture_2d_array<f32>;
@group(0) @binding(2) var pointsTextures: texture_2d_array<f32>;
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba32float, write>;

@group(1) @binding(0) var<uniform> uniforms: vec3<u32>;
@group(1) @binding(1) var<storage> depths: array<f32>;
@group(1) @binding(2) var<storage, read_write> iDepths: array<u32>;
@group(1) @binding(3) var<storage, read_write> iColorX: array<u32>;
@group(1) @binding(4) var<storage, read_write> iColorY: array<u32>;

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

    let numberOfLayers = textureNumLayers(reconstructionTextures);

    // Razpon globin: potrebujemo ga za normalizacijo globine vsake plasti, 
    // da lahko prilagodimo SDF in alpha strategijo glede na to, 
    // kako blizu ali daleč je plast.
    var minDepth = depths[0];
    var maxDepth = depths[0];
    for (var d = 1u; d < numberOfLayers; d++) {
        minDepth = min(minDepth, depths[d]);
        maxDepth = max(maxDepth, depths[d]);
    }
    let depthRange = max(maxDepth - minDepth, 1e-5);

    // Začnemo z najbolj oddaljeno plastjo kot polno osnovo
    // layer 0 = najbolj oddaljena, layer N-1 = najbližja.

    // Začnemo z prozornim ozadjem, nato bomo nanj "prebarvali" vsako plast od zadaj naprej.
    var result = vec4f(0.0, 0.0, 0.0, 0.0);

    for (var i = 0u; i < numberOfLayers; i++) {
        let col = textureLoad(reconstructionTextures, index, i, 0);
        let sdf = textureLoad(sdfTextures, index, i, 0);
        let pts = textureLoad(pointsTextures, index, i, 0);

        let sdfDist = sdf.a;
        let hasData = select(0.0, 1.0, pts.a > 0.01);

        // Kako globoko v sceni je trenutna plast? 0=najbolj oddaljena, 1=najbližja
        let normalizedDepth = (depths[i] - minDepth) / depthRange;
        // Inverz: 0=najbolj oddaljena, 1=najbližja
        let closeness = 1.0 - normalizedDepth;

        // Strategija alpha: zadnje plasti zaupajo rekonstrukciji,
        //    sprednje plasti uporabljajo SDF - uporabijo barve le blizu dejanskih podatkov.
        // Zadnje plasti (closeness ≈ 0)
        //   mora biti popolnoma zapolnjena → visoka vrednost alpha, nežno SDF upadanje
        //
        // Sprednje plasti (closeness ≈ 1)
        //   rekonstrukcija naj pokaže le bližnje dejanske podatke → tesen SDF

        // SDF falloff: back=soft (k=30), front=tight (k=9000)
        let k = mix(30.0, 9000.0, closeness);
        let sdfAlpha = exp(-(sdfDist * sdfDist * k));

        // Max SDF razdalja: back=dovolimo veliko (0.3), front=dovolimo le bližnje (0.15)
        let maxDist = mix(0.2, 0.15, closeness);

        // Base alpha strategija:
        //   zadnje plasti → zaupajo rekonstrukciji skoraj popolnoma
        //   sprednje plasti → zaupajo le bližnjim dejanskim podatkom
        var alpha: f32;
        if (sdfDist > maxDist && hasData < 0.5) {
            alpha = 0.0;
        } else if (hasData > 0.5) {
            // Dejanski podatki točk → visoka alpha
            alpha = 0.95;
        } else {
            // Rekonstruirani piksel: zadnje plasti dobijo visoko alpha, sprednje dobijo SDF-modulirano
            let baseAlpha = mix(0.99, 0.8, closeness);
            alpha = baseAlpha * sdfAlpha;
        }

        if (alpha < 0.005) {
            continue;
        }

        // Back-to-front "over": nova barva se "prebarva" na obstoječo
        result = vec4f(
            mix(result.rgb, col.rgb, alpha),
            result.a + alpha * (1.0 - result.a)
        );
    }

    textureStore(outputTexture, index, result);
}
