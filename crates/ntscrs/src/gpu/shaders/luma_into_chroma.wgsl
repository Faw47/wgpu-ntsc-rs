@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_plane: array<f32>;

struct ShaderParams {
    width: u32,
    frame_num: u32,
    seed_hi: u32,
    seed_lo: u32,

    noise_frequency: f32,
    noise_intensity: f32,
    noise_detail: u32,
    snow_anisotropy: f32,

    phase_shift: u32,
    phase_offset: i32,
    filter_mode: u32,
    chroma_delay_horizontal: f32,

    chroma_delay_vertical: i32,
    horizontal_scale: f32,
    vertical_scale: f32,
    mid_line_position: f32,

    mid_line_enabled: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(1) @binding(0) var<uniform> params: ShaderParams;

fn chroma_phase_shift(line_num: u32, frame_num: u32, phase_shift: u32, phase_offset: i32) -> u32 {
    if (phase_shift == 0u) {
        return 0u;
    } else if (phase_shift == 1u || phase_shift == 3u) {
        return u32((i32(frame_num) + phase_offset + i32(line_num >> 1u)) & 3i);
    } else if (phase_shift == 2u) {
        return u32((i32((frame_num + line_num) & 2u) + phase_offset) & 3i);
    }
    return 0u;
}

// We compute Y manually per-mode to avoid race conditions.
fn get_box_y(index: u32) -> f32 {
    let width = params.width;
    let cur = scratch_plane[index];
    var p0 = 16.0 / 255.0;
    if (index % width > 0u) { p0 = scratch_plane[index - 1u]; }
    var p2 = cur;
    if (index % width < width - 1u) { p2 = scratch_plane[index + 1u]; }
    var p3 = cur;
    if (index % width + 2u < width) { p3 = scratch_plane[index + 2u]; }
    else if (index % width < width - 1u) { p3 = scratch_plane[index + 1u]; }

    return (p0 + cur + p2 + p3) * 0.25;
}

fn get_notch_y(index: u32) -> f32 {
    return y_plane[index]; // Notch runs a separate IIR pass before this, so y_plane is safe
}

fn get_one_line_comb_y(index: u32, width: u32) -> f32 {
    let line_num = index / width;
    let top_idx = select(index - width, index + width, line_num == 0u);
    return (scratch_plane[top_idx] + scratch_plane[index]) * 0.5;
}

fn get_two_line_comb_y(index: u32, width: u32, height: u32) -> f32 {
    let line_num = index / width;
    let prev_idx = i32(index) + select(-(i32(width)), i32(width), line_num == 0u);
    let next_idx = i32(index) + select(i32(width), -(i32(width)), line_num == height - 1u);
    let cur = scratch_plane[index];
    let prev = scratch_plane[u32(prev_idx)];
    let next = scratch_plane[u32(next_idx)];
    return (cur * 0.5) + (prev * 0.25) + (next * 0.25);
}

fn process_chroma(val: f32, shift: u32, index: u32, xi: u32) -> vec2<f32> {
    let offset = ((index % params.width) + shift + xi) & 3u;
    var i_v = val * select(0.5, 1.0, shift == 0u);
    var q_v = val * select(0.5, 1.0, shift == 0u);
    if (offset == 0u) { i_v *= -1.0; q_v *= 0.0; }
    else if (offset == 1u) { i_v *= 0.0; q_v *= -1.0; }
    else if (offset == 2u) { i_v *= 1.0; q_v *= 0.0; }
    else if (offset == 3u) { i_v *= 0.0; q_v *= 1.0; }
    return vec2<f32>(i_v, q_v);
}

fn apply_chroma(index: u32, xi: u32, c: f32, c_left: f32, c_right: f32) {
    var i_val = 0.0;
    var q_val = 0.0;

    let center = process_chroma(c, 0u, index, xi);
    i_val += center.x;
    q_val += center.y;

    let width = params.width;
    if (index % width > 0u) {
        let left = process_chroma(c_left, 4294967295u, index, xi); // -1u
        i_val += left.x;
        q_val += left.y;
    }
    if (index % width < width - 1u) {
        let right = process_chroma(c_right, 1u, index, xi);
        i_val += right.x;
        q_val += right.y;
    }

    i_plane[index] = i_val;
    q_plane[index] = q_val;
}

@compute @workgroup_size(16, 16, 1)
fn demodulate_box(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let row_idx = global_id.y;
    let width = params.width;
    let height = arrayLength(&y_plane) / width;

    if (col_idx >= width || row_idx >= height) {
        return;
    }

    let index = row_idx * width + col_idx;
    let xi = chroma_phase_shift(row_idx * 2u, params.frame_num, params.phase_shift, params.phase_offset);

    let y_c = get_box_y(index);
    var c_l = 0.0;
    var c_r = 0.0;
    if (col_idx > 0u) {
        c_l = get_box_y(index - 1u) - scratch_plane[index - 1u];
    }
    if (col_idx + 1u < width) {
        c_r = get_box_y(index + 1u) - scratch_plane[index + 1u];
    }

    y_plane[index] = y_c;

    let c_c = y_c - scratch_plane[index];

    apply_chroma(index, xi, c_c, c_l, c_r);
}

@compute @workgroup_size(16, 16, 1)
fn demodulate_notch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let row_idx = global_id.y;
    let width = params.width;
    let height = arrayLength(&y_plane) / width;

    if (col_idx >= width || row_idx >= height) {
        return;
    }

    let index = row_idx * width + col_idx;
    let xi = chroma_phase_shift(row_idx * 2u, params.frame_num, params.phase_shift, params.phase_offset);

    let y_c = get_notch_y(index);
    var c_l = 0.0;
    var c_r = 0.0;
    if (col_idx > 0u) {
        c_l = get_notch_y(index - 1u) - scratch_plane[index - 1u];
    }
    if (col_idx + 1u < width) {
        c_r = get_notch_y(index + 1u) - scratch_plane[index + 1u];
    }

    let c_c = y_c - scratch_plane[index];

    apply_chroma(index, xi, c_c, c_l, c_r);
}

@compute @workgroup_size(16, 16, 1)
fn demodulate_one_line_comb(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let row_idx = global_id.y;
    let width = params.width;
    let height = arrayLength(&y_plane) / width;

    if (col_idx >= width || row_idx >= height) {
        return;
    }

    let index = row_idx * width + col_idx;
    let xi = chroma_phase_shift(row_idx * 2u, params.frame_num, params.phase_shift, params.phase_offset);

    let y_c = get_one_line_comb_y(index, width);
    var c_l = 0.0;
    var c_r = 0.0;
    if (col_idx > 0u) {
        c_l = get_one_line_comb_y(index - 1u, width) - scratch_plane[index - 1u];
    }
    if (col_idx + 1u < width) {
        c_r = get_one_line_comb_y(index + 1u, width) - scratch_plane[index + 1u];
    }

    y_plane[index] = y_c;

    let c_c = y_c - scratch_plane[index];

    apply_chroma(index, xi, c_c, c_l, c_r);
}

@compute @workgroup_size(16, 16, 1)
fn demodulate_two_line_comb(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let row_idx = global_id.y;
    let width = params.width;
    let height = arrayLength(&y_plane) / width;

    if (col_idx >= width || row_idx >= height) {
        return;
    }

    let index = row_idx * width + col_idx;
    let xi = chroma_phase_shift(row_idx * 2u, params.frame_num, params.phase_shift, params.phase_offset);

    let y_c = get_two_line_comb_y(index, width, height);
    var c_l = 0.0;
    var c_r = 0.0;
    if (col_idx > 0u) {
        c_l = get_two_line_comb_y(index - 1u, width, height) - scratch_plane[index - 1u];
    }
    if (col_idx + 1u < width) {
        c_r = get_two_line_comb_y(index + 1u, width, height) - scratch_plane[index + 1u];
    }

    y_plane[index] = y_c;

    let c_c = y_c - scratch_plane[index];

    apply_chroma(index, xi, c_c, c_l, c_r);
}
