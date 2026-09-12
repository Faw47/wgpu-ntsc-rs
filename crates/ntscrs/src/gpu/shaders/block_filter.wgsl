// Experimental affine block-boundary decomposition. No production entry point uses this shader.
struct Params {
    num: vec4<f32>, den: vec4<f32>, initial: vec4<f32>, transition: vec4<f32>,
    width: u32, rows: u32, blocks: u32, delay: u32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> summaries: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> boundaries: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> p: Params;
fn next_state(z: vec2<f32>, x: f32, y: f32) -> vec2<f32> {
    return vec2<f32>(p.num.y * x + z.y - p.den.x * y, p.num.z * x - p.den.y * y);
}
@compute @workgroup_size(64)
fn summarize(@builtin(global_invocation_id) id: vec3<u32>) {
    let block = id.x; let row = id.y;
    if block >= p.blocks || row >= p.rows { return; }
    var z = vec2<f32>(0.0);
    for (var i = block * 64u; i < min((block + 1u) * 64u, p.width + p.delay); i++) {
        let x = input[row * p.width + min(i, p.width - 1u)];
        let y = p.num.x * x + z.x;
        z = next_state(z, x, y);
    }
    summaries[row * p.blocks + block] = z;
}
@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= p.rows { return; }
    var z = p.initial.xy * input[row * p.width];
    for (var block = 0u; block < p.blocks; block++) {
        let idx = row * p.blocks + block;
        boundaries[idx] = z;
        z = vec2<f32>(p.transition.x * z.x + p.transition.y * z.y,
                      p.transition.z * z.x + p.transition.w * z.y) + summaries[idx];
    }
}
@compute @workgroup_size(64)
fn replay(@builtin(global_invocation_id) id: vec3<u32>) {
    let block = id.x; let row = id.y;
    if block >= p.blocks || row >= p.rows { return; }
    var z = boundaries[row * p.blocks + block];
    for (var i = block * 64u; i < min((block + 1u) * 64u, p.width + p.delay); i++) {
        let x = input[row * p.width + min(i, p.width - 1u)];
        let y = p.num.x * x + z.x;
        z = next_state(z, x, y);
        if i >= p.delay { output[row * p.width + i - p.delay] = y; }
    }
}
@compute @workgroup_size(64)
fn serial(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if row >= p.rows { return; }
    var z = p.initial.xy * input[row * p.width];
    for (var i = 0u; i < p.width + p.delay; i++) {
        let x = input[row * p.width + min(i, p.width - 1u)];
        let y = p.num.x * x + z.x;
        z = next_state(z, x, y);
        if i >= p.delay { output[row * p.width + i - p.delay] = y; }
    }
}
