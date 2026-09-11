@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> scratch_i: array<f32>;
@group(0) @binding(5) var<storage, read_write> scratch_q: array<f32>;

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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let row_idx = global_id.y;
    let width = params.width;
    let height = arrayLength(&i_plane) / width;

    if (col_idx >= width || row_idx >= height) {
        return;
    }

    let index = row_idx * width + col_idx;

    // Apply vertical offset
    let dst_y = i32(row_idx) - params.chroma_delay_vertical;
    if (dst_y < 0 || dst_y >= i32(height)) {
        i_plane[index] = 0.0;
        q_plane[index] = 0.0;
        return;
    }

    // Apply horizontal offset
    let src_x_float = f32(col_idx) - params.chroma_delay_horizontal;

    // Linear interp for shift
    let src_x_left = i32(floor(src_x_float));
    let src_x_right = src_x_left + 1;
    let fract_x = src_x_float - f32(src_x_left);

    var i_val = 0.0;
    var q_val = 0.0;

    // Left sample
    if (src_x_left >= 0 && src_x_left < i32(width)) {
        let left_idx = u32(dst_y) * width + u32(src_x_left);
        i_val += scratch_i[left_idx] * (1.0 - fract_x);
        q_val += scratch_q[left_idx] * (1.0 - fract_x);
    }

    // Right sample
    if (src_x_right >= 0 && src_x_right < i32(width)) {
        let right_idx = u32(dst_y) * width + u32(src_x_right);
        i_val += scratch_i[right_idx] * fract_x;
        q_val += scratch_q[right_idx] * fract_x;
    }

    i_plane[index] = i_val;
    q_plane[index] = q_val;
}
