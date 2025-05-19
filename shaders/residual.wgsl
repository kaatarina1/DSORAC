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
    
    // Get boundary condition from points texture
    let points = textureLoad(pointsTexture, coord, 0);
    
    // Load current reconstruction value
    let recon = textureLoad(reconstructionTexture, coord, 0);
    
    // Initialize residual
    var residual: vec4<f32>;
    
    // If this is a fixed point (boundary condition)
    if (points.a >= 1.0) {
        // For boundary points, residual is zero
        residual = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    } else {
        // For non-boundary points, compute the Laplacian residual
        let h = 1.0 / f32(size.x - 1u); // Assuming square texture
        
        // Sample neighbors
        var left = vec4<f32>(0.0);
        var right = vec4<f32>(0.0);
        var bottom = vec4<f32>(0.0);
        var top = vec4<f32>(0.0);
        
        if (coord.x > 0) {
            left = textureLoad(reconstructionTexture, vec2<i32>(coord.x - 1, coord.y), 0);
        } else {
            left = recon;
        }
        
        if (coord.x < i32(size.x) - 1) {
            right = textureLoad(reconstructionTexture, vec2<i32>(coord.x + 1, coord.y), 0);
        } else {
            right = recon;
        }
        
        if (coord.y > 0) {
            bottom = textureLoad(reconstructionTexture, vec2<i32>(coord.x, coord.y - 1), 0);
        } else {
            bottom = recon;
        }
        
        if (coord.y < i32(size.y) - 1) {
            top = textureLoad(reconstructionTexture, vec2<i32>(coord.x, coord.y + 1), 0);
        } else {
            top = recon;
        }
        
        // Calculate Laplacian
        let laplacian = (left + right + bottom + top - 4.0 * recon) / (h * h);
        
        // The residual is points - Laplacian of reconstruction
        residual = points - laplacian;
    }
    
    // Store the residual
    textureStore(outputTexture, coord, residual);
    
    // Update residual norm (for convergence checking)
    let residualSum = residual.r * residual.r + residual.g * residual.g + 
                      residual.b * residual.b + residual.a * residual.a;
    atomicAdd(&residualNorm, u32(residualSum * 1000.0)); // Scale for integer conversion
}
