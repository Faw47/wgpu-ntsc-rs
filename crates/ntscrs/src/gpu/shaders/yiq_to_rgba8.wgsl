@group(0) @binding(0) var<storage, read> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> rgba_output: array<u32>;

struct OutputParams {
    width: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(4) var<uniform> output_params: OutputParams;

fn to_u8(value: f32) -> u32 {
    return u32(clamp(value, 0.0, 1.0) * 255.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    let width = output_params.width;
    let height = arrayLength(&y_plane) / width;
    if (col >= width || row >= height) {
        return;
    }

    let index = row * width + col;
    let y = y_plane[index];
    let i = i_plane[index];
    let q = q_plane[index];
    // Match the CPU yiq_to_rgb() evaluation order: each channel is a
    // multiply-add chain with a fused final rounding step in the strict path.
    let r = to_u8(exact_fma_f32(0.619, q, exact_fma_f32(0.956, i, y)));
    let g = to_u8(exact_fma_f32(-0.647, q, exact_fma_f32(-0.272, i, y)));
    let b = to_u8(exact_fma_f32(1.703, q, exact_fma_f32(-1.106, i, y)));
    rgba_output[index] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
