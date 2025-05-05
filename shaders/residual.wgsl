@group(0) @binding(0) var pointsTexture: texture_2d<f32>;
@group(0) @binding(1) var reconstructionTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<storage, read_write> residualNorm: atomic<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let size = textureDimensions(outputTexture);
    
    // Check if within texture bounds
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }

    // Calculate texture coordinates
    let coord = vec2<i32>(global_id.xy);
    let h = 1.0 / f32(size.x); // Assuming square texture
    
    // Sample reconstruction at center and neighbors
    let center = textureLoad(reconstructionTexture, coord, 0);
    
    // Handle boundary conditions for neighbors
    var left = vec4<f32>(0.0);
    var right = vec4<f32>(0.0);
    var bottom = vec4<f32>(0.0);
    var top = vec4<f32>(0.0);
    if (coord.x > 0) {
        left = textureLoad(reconstructionTexture, vec2<i32>(coord.x - 1, coord.y), 0);
    } else {
        left = center;
    }
    if (coord.x < i32(size.x) - 1) {
        right = textureLoad(reconstructionTexture, vec2<i32>(coord.x + 1, coord.y), 0);
    } else {
        right = center;
    }
    if (coord.y > 0) {
        bottom = textureLoad(reconstructionTexture, vec2<i32>(coord.x, coord.y - 1), 0);
    } else {
        bottom = center;
    }
    if (coord.y < i32(size.y) - 1) {
        top = textureLoad(reconstructionTexture, vec2<i32>(coord.x, coord.y + 1), 0);
    } else {
        top = center;
    }

    // Sample points texture to get boundary conditions
    let points = textureLoad(pointsTexture, coord, 0);
    
    // Calculate Laplacian: (left + right + bottom + top - 4*center) / h²
    let laplacian = (left + right + bottom + top - 4.0 * center) / (h * h);
    
    // Calculate residual: f - Laplacian(u)
    // f is zero if not a boundary point, otherwise it's the fixed value
    var residual = vec4<f32>(0.0);
    if (points.a > 0.5) {
        // Boundary condition, use fixed value from points texture
        residual = vec4<f32>(0.0);
    } else {
        // Interior point, f is zero
        residual = -laplacian;
    }
    
    // Update residual norm (using atomic operations for reduction)
    let residualSquared = dot(residual.rgb, residual.rgb);
    let residualInt = u32(residualSquared * 1000.0); // Scale to avoid precision issues
    atomicAdd(&residualNorm, residualInt);
    
    // Store residual in output texture
    textureStore(outputTexture, coord, vec4<f32>(residual.rgb, 1.0));
}