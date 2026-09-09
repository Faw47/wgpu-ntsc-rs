@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_plane: array<f32>;

struct ShaderParams {
    width: u32,
    frame_num: u32,
    seed: u32,
    noise_idx: u32,

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
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}
@group(1) @binding(0) var<uniform> params: ShaderParams;

struct FilterCoeffs {
    num: vec4<f32>,
    den: vec4<f32>,
    z_initial: vec4<f32>,
    delay: u32,
    filter_len: u32,
    plane_idx: u32, // 0: y, 1: i, 2: q
    initial_condition_mode: u32, // 0: use z_initial, 1: FirstSample
}
@group(2) @binding(0) var<uniform> filter_coeffs: FilterCoeffs;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let height = arrayLength(&y_plane) / width;
    
    if (row_idx >= height) {
        return;
    }

    let row_start = row_idx * width;
    var z = filter_coeffs.z_initial;
    let num = filter_coeffs.num;
    let den = filter_coeffs.den;
    let filter_len = filter_coeffs.filter_len;
    let delay = filter_coeffs.delay;
    let plane_idx = filter_coeffs.plane_idx;

    // Matches TransferFunction::initial_condition_into (scipy) for FirstSample, like the CPU path.
    if (filter_coeffs.initial_condition_mode == 1u) {
        var initial_val: f32;
        if (plane_idx == 0u) { initial_val = y_plane[row_start]; }
        else if (plane_idx == 1u) { initial_val = i_plane[row_start]; }
        else if (plane_idx == 2u) { initial_val = q_plane[row_start]; }
        else { initial_val = scratch_plane[row_start]; }

        z *= initial_val;
    }

    // We process width + delay iterations.
    for (var i = 0u; i < width + delay; i++) {
        let read_idx = row_start + min(i, width - 1u);
        var sample_val: f32;
        if (plane_idx == 0u) { sample_val = y_plane[read_idx]; }
        else if (plane_idx == 1u) { sample_val = i_plane[read_idx]; }
        else if (plane_idx == 2u) { sample_val = q_plane[read_idx]; }
        else { sample_val = scratch_plane[read_idx]; }

        let filt_sample = z.x + (num.x * sample_val);

        if (filter_len > 1u) {
            z.x = z.y + (num.y * sample_val) - (den.x * filt_sample);
        }
        if (filter_len > 2u) {
            z.y = z.z + (num.z * sample_val) - (den.y * filt_sample);
        }
        if (filter_len > 3u) {
            z.z = z.w + (num.w * sample_val) - (den.z * filt_sample);
        }

        if (i >= delay) {
            let write_idx = row_start + i - delay;
            if (plane_idx == 0u) { y_plane[write_idx] = filt_sample; }
            else if (plane_idx == 1u) { i_plane[write_idx] = filt_sample; }
            else if (plane_idx == 2u) { q_plane[write_idx] = filt_sample; }
            else { scratch_plane[write_idx] = filt_sample; }
        }
    }
}
