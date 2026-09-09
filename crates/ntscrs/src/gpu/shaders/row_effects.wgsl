@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
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
@group(2) @binding(0) var<storage, read> data: array<u32>;
fn rf(row: u32, field: u32) -> f32 {return bitcast<f32>(data[row * 12u + field]);}
fn ru(row: u32, field: u32) -> u32 {return data[row * 12u + field];}
fn valid(id: vec3<u32>) -> bool { return id.x < params.width && id.y < arrayLength(&y_plane) / params.width; }

@compute @workgroup_size(16, 16, 1)
fn noise(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!valid(id) || rf(id.y, 3u) == 0.0) {return;}
    let index = id.y * params.width + id.x;
    let value = fbm_1d(bitcast<i32>(ru(id.y, 1u)), ru(id.y, 5u), 1.0, 2.0,
        rf(id.y, 4u), f32(id.x) + rf(id.y, 2u));
    y_plane[index] += value * rf(id.y, 3u);
}
fn shifted(id: vec3<u32>, plane: u32) -> f32 {
    let shift = rf(id.y, 0u);
    let base = plane * arrayLength(&y_plane) + id.y * params.width;
    let source = f32(id.x) - shift;
    let left = i32(floor(source));
    let right = left + 1;
    let fraction = source - f32(left);
    var boundary = 0.0;
    if (ru(id.y, 10u) != 0u) {
        boundary = scratch[base + select(params.width - 1u, 0u, shift >= 0.0)];
    }
    var a = boundary;
    var b = boundary;
    if (left >= 0 && left < i32(params.width)) {a = scratch[base + u32(left)];}
    if (right >= 0 && right < i32(params.width)) {b = scratch[base + u32(right)];}
    return a * (1.0 - fraction) + b * fraction;
}
@compute @workgroup_size(16, 16, 1)
fn shift_y(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!valid(id) || id.x < ru(id.y, 7u)) {return;}
    let index = id.y * params.width + id.x;
    var value = shifted(id, 0u);
    let x = f32(id.x - ru(id.y, 7u));
    let length = rf(id.y, 8u);
    if (length > 0.0 && x < ceil(length)) {
        let envelope = 1.0 - x / length;
        value += envelope * envelope * envelope * rf(id.y, 9u);
    }
    y_plane[index] = value;
}
@compute @workgroup_size(16, 16, 1)
fn shift_all(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!valid(id)) {return;}
    let index = id.y * params.width + id.x;
    y_plane[index] = shifted(id, 0u);
    i_plane[index] = shifted(id, 1u);
    q_plane[index] = shifted(id, 2u);
}
@compute @workgroup_size(16, 16, 1)
fn phase(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!valid(id)) {return;}
    let index = id.y * params.width + id.x;
    let i = i_plane[index];
    let q = q_plane[index];
    let sine = rf(id.y, 6u);
    let cosine = rf(id.y, 4u);
    i_plane[index] = i * cosine - q * sine;
    q_plane[index] = i * sine + q * cosine;
}
@compute @workgroup_size(16, 16, 1)
fn loss(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!valid(id) || ru(id.y, 11u) == 0u) {return;}
    let index = id.y * params.width + id.x;
    i_plane[index] = 0.0;
    q_plane[index] = 0.0;
}
@compute @workgroup_size(16, 16, 1)
fn snow(@builtin(global_invocation_id) id: vec3<u32>) {
    if (!valid(id)) {return;}
    let tile = id.y * ((params.width + 31u) / 32u) + id.x / 32u;
    let index = id.y * params.width + id.x;
    var value = y_plane[index];
    for (var cursor = data[4u + tile]; cursor < data[5u + tile]; cursor++) {
        let event = data[0] + data[cursor] * 4u;
        let start = bitcast<i32>(data[event]);
        let length = bitcast<f32>(data[event + 1u]);
        let frequency = bitcast<f32>(data[event + 2u]);
        if (i32(id.x) < start || i32(id.x) >= start + i32(ceil(length))) {continue;}
        let x = f32(i32(id.x) - start);
        let random_idx = data[1] + data[event + 3u] + id.x - u32(max(0, start));
        let random = bitcast<f32>(data[random_idx]);
        let envelope = 1.0 - x / length;
        value += cos((x * 3.141592653589793) / frequency) * (envelope * envelope) * random;
    }
    y_plane[index] = value;
}
