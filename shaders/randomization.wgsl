struct Particle {
    position: vec3f,
    color: u32
}

@group(0) @binding(0) var<storage, read> positions: array<vec3f>;
@group(0) @binding(1) var<storage, read> colors: array<u32>;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(3) var<uniform> numberOfParticles: u32;

@compute
@workgroup_size(256)
fn compute(@builtin(global_invocation_id) globalId: vec3u) {
    let index = globalId.x;
    if (index >= numberOfParticles) {
        return;
    }

    let position = positions[index];
    let color = colors[index];

    particles[index] = Particle(position, color);
    // Debug: write index to particle position
    // particles[index] = Particle(vec3f(f32(index), 0.0, 0.0), color);
}
