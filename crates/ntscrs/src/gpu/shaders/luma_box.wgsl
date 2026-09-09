@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
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
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.width;
    if (width < 2u || id.x >= arrayLength(&y_plane) / width) { return; }
    let start = id.x * width;
    var delay = array<f32, 4>(16.0 / 255.0, 16.0 / 255.0, y_plane[start], y_plane[start + 1u]);
    var sum = ((delay[0] + delay[1]) + delay[2]) + delay[3];
    let last = y_plane[start + width - 1u];
    for (var x = 0u; x < width; x++) {
        var value = last;
        if (x + 2u < width) { value = y_plane[start + x + 2u]; }
        sum -= delay[x % 4u];
        delay[x % 4u] = value;
        sum += value;
        y_plane[start + x] = sum * 0.25;
    }
}
