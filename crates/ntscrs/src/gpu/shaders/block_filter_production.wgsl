// Native-f32 affine block IIR filter used by the release GPU path. The input
// is copied to the corresponding scratch plane before dispatch, so the output
// plane can be written in place without read/write aliasing between blocks.
@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> scratch_i: array<f32>;
@group(0) @binding(5) var<storage, read_write> scratch_q: array<f32>;

struct Params {
    num: vec4<f32>,
    den: vec4<f32>,
    initial: vec4<f32>,
    transition: vec4<f32>,
    num_q: vec4<f32>,
    den_q: vec4<f32>,
    initial_q: vec4<f32>,
    transition_q: vec4<f32>,
    width: u32,
    rows: u32,
    blocks: u32,
    delay: u32,
    delay_q: u32,
    plane_idx: u32, // 0: y, 1: i, 2: q, 3: i and q together
    filter_len: u32,
    initial_condition_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(1) @binding(0) var<uniform> p: Params;
@group(2) @binding(0) var<storage, read_write> summaries: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> boundaries: array<vec4<f32>>;

fn filter_sample(z: vec2<f32>, sample: f32, num: vec4<f32>, den: vec4<f32>) -> vec3<f32> {
    let y = fma(num.x, sample, z.x);
    let next_x = fma(num.y, sample, z.y) - den.x * y;
    let next_y = fma(num.z, sample, 0.0) - den.y * y;
    return vec3<f32>(y, next_x, next_y);
}

fn read_single(index: u32) -> f32 {
    if (p.plane_idx == 0u) { return scratch_y[index]; }
    if (p.plane_idx == 1u) { return scratch_i[index]; }
    return scratch_q[index];
}

fn write_single(index: u32, value: f32) {
    if (p.plane_idx == 0u) {
        y_plane[index] = value;
    } else if (p.plane_idx == 1u) {
        i_plane[index] = value;
    } else {
        q_plane[index] = value;
    }
}

fn compose(z: vec2<f32>, summary: vec2<f32>, transition: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(
        fma(transition.x, z.x, transition.y * z.y) + summary.x,
        fma(transition.z, z.x, transition.w * z.y) + summary.y,
    );
}

@compute @workgroup_size(64)
fn summarize(@builtin(global_invocation_id) id: vec3<u32>) {
    let block = id.x;
    let row = id.y;
    if (block >= p.blocks || row >= p.rows) { return; }
    let row_start = row * p.width;
    let begin = block * 64u;
    let end = min(begin + 64u, p.width + max(p.delay, p.delay_q));
    if (p.plane_idx == 3u) {
        var zi = vec2<f32>(0.0);
        var zq = vec2<f32>(0.0);
        for (var i = begin; i < end; i++) {
            let index = row_start + min(i, p.width - 1u);
            let yi = filter_sample(zi, scratch_i[index], p.num, p.den);
            let yq = filter_sample(zq, scratch_q[index], p.num_q, p.den_q);
            zi = yi.yz;
            zq = yq.yz;
        }
        summaries[row * p.blocks + block] = vec4<f32>(zi, zq);
    } else {
        var z = vec2<f32>(0.0);
        for (var i = begin; i < end; i++) {
            let index = row_start + min(i, p.width - 1u);
            let result = filter_sample(z, read_single(index), p.num, p.den);
            z = result.yz;
        }
        summaries[row * p.blocks + block] = vec4<f32>(z, 0.0, 0.0);
    }
}

@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= p.rows) { return; }
    let row_start = row * p.width;
    if (p.plane_idx == 3u) {
        var zi = vec2<f32>(0.0);
        var zq = vec2<f32>(0.0);
        if ((p.initial_condition_mode & 1u) != 0u) {
            zi = p.initial.xy * scratch_i[row_start];
            zq = p.initial_q.xy * scratch_q[row_start];
        }
        for (var block = 0u; block < p.blocks; block++) {
            let index = row * p.blocks + block;
            boundaries[index] = vec4<f32>(zi, zq);
            zi = compose(zi, summaries[index].xy, p.transition);
            zq = compose(zq, summaries[index].zw, p.transition_q);
        }
    } else {
        var z = vec2<f32>(0.0);
        if ((p.initial_condition_mode & 1u) != 0u) {
            z = p.initial.xy * read_single(row_start);
        }
        for (var block = 0u; block < p.blocks; block++) {
            let index = row * p.blocks + block;
            boundaries[index] = vec4<f32>(z, 0.0, 0.0);
            z = compose(z, summaries[index].xy, p.transition);
        }
    }
}

@compute @workgroup_size(64)
fn replay(@builtin(global_invocation_id) id: vec3<u32>) {
    let block = id.x;
    let row = id.y;
    if (block >= p.blocks || row >= p.rows) { return; }
    let row_start = row * p.width;
    let begin = block * 64u;
    let end = min(begin + 64u, p.width + max(p.delay, p.delay_q));
    let boundary = boundaries[row * p.blocks + block];
    if (p.plane_idx == 3u) {
        var zi = boundary.xy;
        var zq = boundary.zw;
        for (var i = begin; i < end; i++) {
            let input_index = row_start + min(i, p.width - 1u);
            let yi = filter_sample(zi, scratch_i[input_index], p.num, p.den);
            let yq = filter_sample(zq, scratch_q[input_index], p.num_q, p.den_q);
            zi = yi.yz;
            zq = yq.yz;
            if (i >= p.delay && i < p.width + p.delay) {
                let output_index = row_start + i - p.delay;
                i_plane[output_index] = yi.x;
            }
            if (i >= p.delay_q && i < p.width + p.delay_q) {
                let output_index = row_start + i - p.delay_q;
                q_plane[output_index] = yq.x;
            }
        }
    } else {
        var z = boundary.xy;
        for (var i = begin; i < end; i++) {
            let input_index = row_start + min(i, p.width - 1u);
            let result = filter_sample(z, read_single(input_index), p.num, p.den);
            z = result.yz;
            if (i >= p.delay) {
                write_single(row_start + i - p.delay, result.x);
            }
        }
    }
}
