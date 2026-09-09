@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;

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

/// Full turn in radians; matches chroma_phase_offset_line (offset * 2π).
const TAU: f32 = 6.28318530718;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col_idx = global_id.x;
    let row_idx = global_id.y;
    let width = params.width;
    let height = arrayLength(&i_plane) / width;

    if (col_idx >= width || row_idx >= height) {
        return;
    }

    let idx = row_idx * width + col_idx;
    var i_val = i_plane[idx];
    var q_val = q_plane[idx];

    // noise_frequency = chroma_phase_error (normalized); noise_intensity = chroma_phase_noise_intensity.
    var total_angle = params.noise_frequency * TAU;

    if (params.noise_intensity > 0.0) {
        let seed_val = vec2<u32>(params.seed, params.noise_idx);
        var h = u64_add(seed_val, vec2<u32>(0u, row_idx));
        h = u64_xor(h, u64_shr(h, 33u));
        h = u64_mul(h, vec2<u32>(0xff51afd7u, 0xed558ccdu));
        h = u64_xor(h, u64_shr(h, 33u));
        h = u64_mul(h, vec2<u32>(0xc4ceb9feu, 0x1a85ec53u));
        h = u64_xor(h, u64_shr(h, 33u));

        let val = f32(h.y) / 4294967296.0;
        // Same as ntsc.rs chroma_phase_noise -> chroma_phase_offset_line (× 2π).
        let phase_noise = (val * 2.0 - 1.0) * params.noise_intensity * TAU;
        total_angle += phase_noise;
    }

    if (total_angle != 0.0) {
        let sin_p = sin(total_angle);
        let cos_p = cos(total_angle);
        let new_i = i_val * cos_p - q_val * sin_p;
        let new_q = i_val * sin_p + q_val * cos_p;
        i_plane[idx] = new_i;
        q_plane[idx] = new_q;
    }
}
