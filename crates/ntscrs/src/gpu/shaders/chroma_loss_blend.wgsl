@group(0) @binding(0) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> q_plane: array<f32>;

struct Params {
    width: u32,
    frame_num: u32,
    seed: u32,
    noise_idx: u32,
    
    noise_frequency: f32, // intensity for chroma_loss
    noise_intensity: f32,
    noise_detail: u32,
    snow_anisotropy: f32,

    phase_shift: u32,
    phase_offset: i32,
    filter_mode: u32,
    chroma_delay_horizontal: f32,

    chroma_delay_vertical: i32,
    horizontal_scale: f32,
    _pad1: u32,
    _pad2: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn chroma_vert_blend(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 1 thread per column to sequentially blend vertically
    let col_idx = global_id.x;
    let width = params.width;
    let height = arrayLength(&i_plane) / width;
    
    if (col_idx >= width) { return; }
    
    var delay_i = 0.0;
    var delay_q = 0.0;
    
    for (var y = 0u; y < height; y++) {
        let idx = y * width + col_idx;
        
        let c_i = i_plane[idx];
        let c_q = q_plane[idx];
        
        i_plane[idx] = (delay_i + c_i) * 0.5;
        q_plane[idx] = (delay_q + c_q) * 0.5;
        
        delay_i = c_i;
        delay_q = c_q;
    }
}
