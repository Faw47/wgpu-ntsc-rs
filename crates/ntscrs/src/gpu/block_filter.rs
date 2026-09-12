//! Experimental low-order IIR block decomposition, deliberately outside the production runner.
//! A block maps incoming state s to A^64 s + b. Summarize, propagate boundaries, replay.
//! This preserves the recurrence mathematically, but does not preserve float evaluation order.
use super::wgpu_backend::{WgpuBackend, WgpuBackendError};
pub use crate::filter::TransferFunction;
use wgpu::util::DeviceExt;

pub const BLOCK_SIZE: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    num: [f32; 4],
    den: [f32; 4],
    initial: [f32; 4],
    transition: [f32; 4],
    width: u32,
    rows: u32,
    blocks: u32,
    delay: u32,
}

fn step(p: &Params, z: &mut [f32; 2], x: f32) -> f32 {
    let y = p.num[0] * x + z[0];
    *z = [
        p.num[1] * x + z[1] - p.den[0] * y,
        p.num[2] * x - p.den[1] * y,
    ];
    y
}

fn params(
    tf: &TransferFunction,
    width: usize,
    rows: usize,
    delay: usize,
) -> Result<Params, String> {
    if !(2..=3).contains(&tf.len()) {
        return Err("prototype supports first-order and biquad filters only".into());
    }
    if width == 0
        || rows == 0
        || width
            .checked_add(delay)
            .is_none_or(|n| n > u32::MAX as usize)
        || rows > u32::MAX as usize
    {
        return Err("invalid experimental filter dimensions".into());
    }
    let (num, den, initial) = tf.to_gpu_coeffs(1.0);
    let mut p = Params {
        num,
        den,
        initial,
        transition: [0.0; 4],
        width: width as u32,
        rows: rows as u32,
        blocks: (width + delay).div_ceil(BLOCK_SIZE) as u32,
        delay: delay as u32,
    };
    // Columns of the block transition, computed with the same recurrence as the samples.
    let mut a = [1.0, 0.0];
    let mut b = [0.0, 1.0];
    for _ in 0..BLOCK_SIZE {
        step(&p, &mut a, 0.0);
        step(&p, &mut b, 0.0);
    }
    p.transition = [a[0], b[0], a[1], b[1]];
    Ok(p)
}

pub fn cpu_block_filter(
    tf: &TransferFunction,
    input: &[f32],
    delay: usize,
    initial: f32,
) -> Result<Vec<f32>, String> {
    let p = params(tf, input.len(), 1, delay)?;
    let mut summaries = vec![[0.0; 2]; p.blocks as usize];
    for (block, z) in summaries.iter_mut().enumerate() {
        for i in block * BLOCK_SIZE..((block + 1) * BLOCK_SIZE).min(input.len() + delay) {
            step(&p, z, input[i.min(input.len() - 1)]);
        }
    }
    let mut boundaries = vec![[0.0; 2]; summaries.len()];
    let mut z = [p.initial[0] * initial, p.initial[1] * initial];
    for (block, b) in summaries.iter().enumerate() {
        boundaries[block] = z;
        z = [
            p.transition[0] * z[0] + p.transition[1] * z[1] + b[0],
            p.transition[2] * z[0] + p.transition[3] * z[1] + b[1],
        ];
    }
    let mut out = vec![0.0; input.len()];
    for (block, mut z) in boundaries.into_iter().enumerate() {
        for i in block * BLOCK_SIZE..((block + 1) * BLOCK_SIZE).min(input.len() + delay) {
            let y = step(&p, &mut z, input[i.min(input.len() - 1)]);
            if i >= delay {
                out[i - delay] = y;
            }
        }
    }
    Ok(out)
}

/// Independent experiment. Resource creation is outside timed dispatches.
/// Input and output never alias, so delayed writes cannot corrupt neighboring blocks.
pub struct BlockFilterExperiment {
    group: wgpu::BindGroup,
    pipelines: [wgpu::ComputePipeline; 4],
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    blocks: u32,
    rows: u32,
    bytes: u64,
}

impl BlockFilterExperiment {
    pub fn new(
        backend: &WgpuBackend,
        tf: &TransferFunction,
        input: &[f32],
        width: usize,
        delay: usize,
        first_sample: bool,
    ) -> Result<Self, WgpuBackendError> {
        if width == 0 || input.len() % width != 0 {
            return Err(WgpuBackendError::Runtime(
                "invalid experiment input shape".into(),
            ));
        }
        let rows = input.len() / width;
        let mut p = params(tf, width, rows, delay).map_err(WgpuBackendError::Runtime)?;
        if !first_sample {
            p.initial = [0.0; 4];
        }
        backend.frame_capacity_requirements(width, rows)?;
        let limits = backend.device.limits();
        if p.blocks.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || p.rows > limits.max_compute_workgroups_per_dimension
        {
            return Err(WgpuBackendError::Runtime(
                "experimental dispatch exceeds device limits".into(),
            ));
        }
        let bytes = input.len() as u64 * 4;
        let summary_bytes = u64::from(p.blocks) * u64::from(p.rows) * 8;
        if summary_bytes
            > u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size)
        {
            return Err(WgpuBackendError::Runtime(
                "experimental boundary storage exceeds device limits".into(),
            ));
        }
        let device = &backend.device;
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("block filter input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = buffer(
            "block filter output",
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let summary = buffer(
            "block summaries",
            summary_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let boundaries = buffer(
            "block boundaries",
            summary_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("block filter params"),
            contents: bytemuck::bytes_of(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("block experiment layout"),
            entries: &std::array::from_fn::<_, 5, _>(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 4 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: i == 0 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block experiment data"),
            layout: &layout,
            entries: &[&input, &output, &summary, &boundaries, &uniform]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("block IIR experiment"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/block_filter.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("block IIR layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipelines = ["summarize", "propagate", "replay", "serial"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let readback = buffer(
            "block experiment readback",
            bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        Ok(Self {
            group,
            pipelines,
            output,
            readback,
            blocks: p.blocks,
            rows: p.rows,
            bytes,
        })
    }

    /// Encode either the parallel experiment or its native-float serial control.
    /// Neither is the integer-emulated production filter.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, parallel: bool) {
        let stages: &[usize] = if parallel { &[0, 1, 2] } else { &[3] };
        for &stage in stages {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines[stage]);
            pass.set_bind_group(0, &self.group, &[]);
            if stage == 1 || stage == 3 {
                pass.dispatch_workgroups(self.rows.div_ceil(64), 1, 1);
            } else {
                pass.dispatch_workgroups(self.blocks.div_ceil(64), self.rows, 1);
            }
        }
    }

    pub fn read(&self, backend: &WgpuBackend) -> Result<Vec<f32>, WgpuBackendError> {
        let mut encoder = backend.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, self.bytes);
        backend.queue.submit(Some(encoder.finish()));
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let result = (|| {
            backend
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| WgpuBackendError::Readback(e.to_string()))?;
            rx.recv()
                .map_err(|e| WgpuBackendError::Readback(e.to_string()))?
                .map_err(|e| WgpuBackendError::Readback(e.to_string()))?;
            Ok(bytemuck::cast_slice(&slice.get_mapped_range()).to_vec())
        })();
        self.readback.unmap();
        result
    }
}
