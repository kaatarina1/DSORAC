@group(0) @binding(0) var reconstructionTextures: texture_2d_array<f32>;
@group(0) @binding(1) var sdfTextures: texture_2d_array<f32>;
@group(0) @binding(2) var pointsTextures: texture_2d_array<f32>;
@group(0) @binding(3) var outputTexture: texture_storage_2d<rgba32float, write>;

@group(1) @binding(0) var<uniform> uniforms: vec3<u32>;
@group(1) @binding(2) var<storage> depths: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size1 = textureDimensions(reconstructionTextures);
    let size2 = textureDimensions(sdfTextures);
    let size3 = textureDimensions(pointsTextures);
    let depth0 = depths[0];
    if (global_id.x >= uniforms.x || global_id.y >= uniforms.y) {
        return;
    }
    let coord = vec2<i32>(global_id.xy);    
    let maxDepth = uniforms.z;

    // for (var i = 0u; i < maxDepth; i = i + 1u) {
    //     let point = textureLoad(pointsTextures, coord, i, 0);
    //     let temp = textureLoad(reconstructionTextures, coord, i, 0);
    //     if (point.a != 0.0) {
    //         textureStore(outputTexture, coord, temp * 255.0);
    //         return;
    //     }
    // }

    let alpha = 0.3;
    let betha = 0.7;

    let threshold = 0.08;
    var previousDist = 0.0;
    var previousAlpha = 0.0;


    var weight = 0.0;
    var newColor = vec4f(0.0, 0.0, 0.0, 0.0);
    for (var i = 0u; i < maxDepth; i = i + 1u) {
        let distance = textureLoad(sdfTextures, coord, i, 0);
        var color = textureLoad(reconstructionTextures, coord, i, 0);

        if (distance.a < threshold) {
            // let normDist = distance.a / threshold;
            // let e = -1.0 * normDist;
            // let weight = exp(e);
            // let colorAlpha = color.a * weight;
            // color = vec4f(color.rgb, colorAlpha);
            // let sumDist = (1 - previousDist) + (1 - distance.a);
            // newColor = (betha * color.a + alpha * (distance.a / sumDist)) * color +  (betha * (1 - color.a) + alpha * previousDist / sumDist) * newColor;
            // previousDist = distance.a;

            let w = exp((distance.a * 1 / log(depths[i])) * 1000.0);
            newColor += color * w;
            weight += w;
            // Distance-based opacity
            // let opacity = exp(-distance.a * 10.0);
    
            // // Depth-based attenuation (closer objects more opaque)
            // let depthFactor = 1.0 / (1.0 + depths[i] * 0.1);
            
            // let finalOpacity = opacity * depthFactor;
            // let weightedColor = color * finalOpacity;
            
            // // Standard alpha blending
            // newColor = newColor * (1.0 - finalOpacity) + weightedColor;
        }
    }

    newColor /= weight;
    newColor = vec4f(newColor.rgb, 1);

    textureStore(outputTexture, vec2<i32>(global_id.xy), newColor * 255);
}
            
        