
// Second shader for final reduction - to be used in a separate dispatch
@group(0) @binding(0) var<storage, read> inputSums: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> finalResult: vec4<f32>;
@group(0) @binding(2) var<uniform> numWorkgroups: vec3<u32>;

@compute @workgroup_size(64)
fn finalReduction(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    // Shared memory for final reduction
    var<workgroup> localData: array<vec4<f32>, 64>;
    
    // Initialize
    let total_workgroups = numWorkgroups.x * numWorkgroups.y;
    
    // Local load with global stride
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    // Each thread processes multiple elements with a stride
    for (var i = local_index; i < total_workgroups; i += 64u) {
        if (i < total_workgroups) {
            sum += inputSums[i];
        }
    }
    
    // Store in local memory
    localData[local_index] = sum;
    workgroupBarrier();
    
    // Final reduction
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (local_index < stride) {
            localData[local_index] += localData[local_index + stride];
        }
        workgroupBarrier();
    }
    
    // Thread 0 writes the final result
    if (local_index == 0u) {
        finalResult = localData[0];
    }
}