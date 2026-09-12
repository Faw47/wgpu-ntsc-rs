use crate::{
    gpu::{GpuBackend, GpuFrame},
    noise_seeds,
    settings::standard::NtscEffect,
    yiq_fielding::YiqView,
};

use std::sync::Arc;
use wgpu::util::DeviceExt;

const IMAGE_STORAGE_BINDINGS: u32 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameCapacityRequirements {
    pub width: usize,
    pub rows: usize,
    pub pixels: u64,
    pub plane_bytes: u64,
    /// Size the former combined scratch representation would have required.
    pub combined_plane_bytes: u64,
}

impl FrameCapacityRequirements {
    pub fn new(width: usize, rows: usize) -> Result<Self, WgpuBackendError> {
        if width == 0 || rows == 0 {
            return Err(WgpuBackendError::InvalidDimensions { width, rows });
        }
        let pixels = u64::try_from(width)
            .ok()
            .and_then(|width| {
                u64::try_from(rows)
                    .ok()
                    .and_then(|rows| width.checked_mul(rows))
            })
            .ok_or(WgpuBackendError::SizeOverflow { width, rows })?;
        let plane_bytes = pixels
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or(WgpuBackendError::SizeOverflow { width, rows })?;
        let combined_plane_bytes = plane_bytes
            .checked_mul(3)
            .ok_or(WgpuBackendError::SizeOverflow { width, rows })?;
        Ok(Self {
            width,
            rows,
            pixels,
            plane_bytes,
            combined_plane_bytes,
        })
    }

    pub fn validate(self, limits: &wgpu::Limits) -> Result<(), WgpuBackendError> {
        if self.plane_bytes > u64::from(limits.max_storage_buffer_binding_size) {
            return Err(WgpuBackendError::UnsupportedCapacity {
                resource: "image plane storage binding",
                required: self.plane_bytes,
                supported: u64::from(limits.max_storage_buffer_binding_size),
            });
        }
        if self.plane_bytes > limits.max_buffer_size {
            return Err(WgpuBackendError::UnsupportedCapacity {
                resource: "image plane buffer",
                required: self.plane_bytes,
                supported: limits.max_buffer_size,
            });
        }
        if limits.max_storage_buffers_per_shader_stage < IMAGE_STORAGE_BINDINGS {
            return Err(WgpuBackendError::UnsupportedCapacity {
                resource: "storage buffers per shader stage",
                required: u64::from(IMAGE_STORAGE_BINDINGS),
                supported: u64::from(limits.max_storage_buffers_per_shader_stage),
            });
        }
        let width_workgroups = u64::try_from(self.width).unwrap_or(u64::MAX).div_ceil(16);
        let row_workgroups = u64::try_from(self.rows).unwrap_or(u64::MAX);
        let supported = u64::from(limits.max_compute_workgroups_per_dimension);
        if width_workgroups > supported {
            return Err(WgpuBackendError::UnsupportedCapacity {
                resource: "horizontal compute workgroups",
                required: width_workgroups,
                supported,
            });
        }
        if row_workgroups > supported {
            return Err(WgpuBackendError::UnsupportedCapacity {
                resource: "row compute workgroups",
                required: row_workgroups,
                supported,
            });
        }
        Ok(())
    }
}

fn storage_buffer_capacity(required: u64, limits: &wgpu::Limits) -> Result<u64, WgpuBackendError> {
    let supported = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
    if required > supported {
        return Err(WgpuBackendError::UnsupportedCapacity {
            resource: "control-data storage binding",
            required,
            supported,
        });
    }
    Ok(required
        .checked_next_power_of_two()
        .unwrap_or(required)
        .min(supported))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WgpuBackendError {
    Initialization(String),
    InvalidDimensions {
        width: usize,
        rows: usize,
    },
    SizeOverflow {
        width: usize,
        rows: usize,
    },
    UnsupportedCapacity {
        resource: &'static str,
        required: u64,
        supported: u64,
    },
    Runtime(String),
    Readback(String),
    DeviceLost(String),
}

impl std::fmt::Display for WgpuBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initialization(message) => write!(f, "WGPU initialization failed: {message}"),
            Self::InvalidDimensions { width, rows } => {
                write!(f, "invalid WGPU frame dimensions {width}x{rows}")
            }
            Self::SizeOverflow { width, rows } => {
                write!(f, "WGPU frame size overflow for {width}x{rows}")
            }
            Self::UnsupportedCapacity {
                resource,
                required,
                supported,
            } => write!(
                f,
                "WGPU {resource} requires {required} bytes/units, device supports {supported}"
            ),
            Self::Runtime(message) => write!(f, "WGPU runtime failure: {message}"),
            Self::Readback(message) => write!(f, "WGPU readback failure: {message}"),
            Self::DeviceLost(message) => write!(f, "WGPU device lost: {message}"),
        }
    }
}

impl std::error::Error for WgpuBackendError {}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParams {
    pub width: u32,
    pub frame_num: u32,
    pub seed: u32,
    pub noise_idx: u32,

    pub noise_frequency: f32,
    pub noise_intensity: f32,
    pub noise_detail: u32,
    pub snow_anisotropy: f32,

    pub phase_shift: u32,
    pub phase_offset: i32,
    pub filter_mode: u32,
    pub chroma_delay_horizontal: f32,

    pub chroma_delay_vertical: i32,
    pub horizontal_scale: f32,
    pub vertical_scale: f32,
    pub _pad1: u32,
    pub _pad2: u32,
    /// Uniform block size must match WGSL `uniform` layout (rounded up to 16-byte boundary).
    pub _pad3: u32,
    pub _pad4: u32,
    pub _pad5: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FilterCoeffs {
    pub num: [f32; 4],
    pub den: [f32; 4],
    pub z_initial: [f32; 4],
    pub delay: u32,
    pub filter_len: u32,
    pub plane_idx: u32,
    pub initial_condition_mode: u32,
}

pub struct WgpuFrame {
    pub y_buffer: wgpu::Buffer,
    pub i_buffer: wgpu::Buffer,
    pub q_buffer: wgpu::Buffer,
    pub scratch_buffers: [wgpu::Buffer; 3],
    pub staging_buffers: [wgpu::Buffer; 3],
    pub main_bind_group: wgpu::BindGroup,
    pub i_pass_bind_group: wgpu::BindGroup,
    pub q_pass_bind_group: wgpu::BindGroup,
    pub chroma_loss_bind_group: wgpu::BindGroup,
    pub width: usize,
    pub height: usize,
    pub full_height: usize,
    // Keep reference to the device/queue to easily do readbacks
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    runtime_errors: Arc<std::sync::Mutex<Vec<String>>>,
    device_lost: Arc<std::sync::Mutex<Option<String>>>,
}

pub struct PendingReadback {
    receivers: [std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>; 3],
}

fn filter_arithmetic_source() -> &'static str {
    let source = include_str!("shaders/filter_plane.wgsl");
    let start = source
        .find("// A four-word unsigned integer")
        .expect("filter arithmetic start marker");
    let end = source
        .find("@compute")
        .expect("filter arithmetic end marker");
    &source[start..end]
}

impl GpuFrame for WgpuFrame {
    fn download(&self, dst: &mut YiqView) {
        self.finish_download(dst, self.enqueue_download());
    }
}

impl WgpuFrame {
    /// Schedule a readback without waiting, allowing independent field work to overlap.
    pub fn enqueue_download(&self) -> PendingReadback {
        let size = (self.width * self.height * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("download encoder"),
            });

        for (source, staging) in [&self.y_buffer, &self.i_buffer, &self.q_buffer]
            .into_iter()
            .zip(&self.staging_buffers)
        {
            encoder.copy_buffer_to_buffer(source, 0, staging, 0, size);
        }

        self.queue.submit(Some(encoder.finish()));

        let receivers = std::array::from_fn(|plane| {
            let buffer_slice = self.staging_buffers[plane].slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                let _ = sender.send(v);
            });
            receiver
        });

        PendingReadback { receivers }
    }

    pub fn try_finish_download(
        &self,
        dst: &mut YiqView,
        pending: PendingReadback,
    ) -> Result<(), WgpuBackendError> {
        if (self.width, self.height) != (dst.dimensions.0, dst.num_rows()) {
            return Err(WgpuBackendError::Readback(format!(
                "destination is {}x{}, frame is {}x{}",
                dst.dimensions.0,
                dst.num_rows(),
                self.width,
                self.height
            )));
        }
        if let Err(error) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            self.staging_buffers.iter().for_each(wgpu::Buffer::unmap);
            if let Some(message) = self
                .device_lost
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
            {
                return Err(WgpuBackendError::DeviceLost(message));
            }
            return Err(WgpuBackendError::Runtime(error.to_string()));
        }
        let mut mapping_error = None;
        for receiver in pending.receivers {
            let result = receiver
                .recv()
                .map_err(|_| WgpuBackendError::Readback("mapping callback was dropped".to_owned()));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    mapping_error = Some(WgpuBackendError::Readback(error.to_string()));
                }
                Err(error) => mapping_error = Some(error),
            }
        }
        if let Some(error) = self
            .device_lost
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
        {
            self.staging_buffers.iter().for_each(wgpu::Buffer::unmap);
            return Err(WgpuBackendError::DeviceLost(error));
        }
        if let Some(error) = mapping_error {
            self.staging_buffers.iter().for_each(wgpu::Buffer::unmap);
            return Err(error);
        }
        let runtime_errors = {
            let mut errors = self
                .runtime_errors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            std::mem::take(&mut *errors)
        };
        if !runtime_errors.is_empty() {
            self.staging_buffers.iter().for_each(wgpu::Buffer::unmap);
            return Err(WgpuBackendError::Runtime(runtime_errors.join("; ")));
        }
        let size = (self.width * self.height * std::mem::size_of::<f32>()) as u64;
        for (staging, destination) in
            self.staging_buffers
                .iter()
                .zip([&mut *dst.y, &mut *dst.i, &mut *dst.q])
        {
            let data = staging.slice(..size).get_mapped_range();
            let source: &[f32] = bytemuck::cast_slice(&data);
            destination[..source.len()].copy_from_slice(source);
        }
        self.staging_buffers.iter().for_each(wgpu::Buffer::unmap);
        Ok(())
    }

    pub fn finish_download(&self, dst: &mut YiqView, pending: PendingReadback) {
        self.try_finish_download(dst, pending)
            .expect("WGPU readback failed");
    }
}

pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    chroma_into_luma_pipeline: wgpu::ComputePipeline,
    luma_into_chroma_box_pipeline: wgpu::ComputePipeline,
    luma_into_chroma_notch_pipeline: wgpu::ComputePipeline,
    luma_into_chroma_one_line_pipeline: wgpu::ComputePipeline,
    luma_into_chroma_two_line_pipeline: wgpu::ComputePipeline,
    luma_box_pipeline: wgpu::ComputePipeline,
    chroma_phase_pipeline: wgpu::ComputePipeline,
    chroma_delay_pipeline: wgpu::ComputePipeline,
    filter_plane_pipeline: wgpu::ComputePipeline,
    chroma_vert_blend_pipeline: wgpu::ComputePipeline,
    filter_coeffs_bind_group_layout: wgpu::BindGroupLayout,
    params_ring_buffer: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    row_layout: wgpu::BindGroupLayout,
    row_pipelines: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
    data_cache: std::cell::RefCell<Vec<(wgpu::Buffer, wgpu::BindGroup, u64)>>,
    data_index: std::cell::Cell<usize>,
    filter_cache: std::cell::RefCell<std::collections::HashMap<[u32; 16], wgpu::BindGroup>>,
    pub adapter_info: wgpu::AdapterInfo,
    pub adapter_limits: wgpu::Limits,
    pub requested_limits: wgpu::Limits,
    simd_lane_count: u32,
    simd_mul_add_fused: bool,
    runtime_errors: Arc<std::sync::Mutex<Vec<String>>>,
    pub(crate) device_lost: Arc<std::sync::Mutex<Option<String>>>,
    dispatch_stages: std::cell::RefCell<Vec<&'static str>>,
    cpu_control_stages: std::cell::RefCell<Vec<&'static str>>,
    pending_error: std::cell::RefCell<Option<WgpuBackendError>>,
    submitted_effects: u64,
}

impl WgpuBackend {
    /// Reuse frame buffers and bind groups across frames of the same dimensions.
    pub fn upload_into(&self, src: &YiqView, frame: &mut WgpuFrame) {
        assert_eq!(
            (src.dimensions.0, src.num_rows()),
            (frame.width, frame.height)
        );
        frame.full_height = src.dimensions.1;
        self.queue
            .write_buffer(&frame.y_buffer, 0, bytemuck::cast_slice(src.y));
        self.queue
            .write_buffer(&frame.i_buffer, 0, bytemuck::cast_slice(src.i));
        self.queue
            .write_buffer(&frame.q_buffer, 0, bytemuck::cast_slice(src.q));
    }

    pub fn new() -> Option<Self> {
        Self::try_new().ok()
    }

    pub fn try_new() -> Result<Self, WgpuBackendError> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Result<Self, WgpuBackendError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| WgpuBackendError::Initialization(error.to_string()))?;

        let adapter_limits = adapter.limits();
        let mut requested_limits = wgpu::Limits::default();
        requested_limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        requested_limits.max_buffer_size = adapter_limits.max_buffer_size;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ntsc-rs wgpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: requested_limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|error| WgpuBackendError::Initialization(error.to_string()))?;

        let runtime_errors = Arc::new(std::sync::Mutex::new(Vec::new()));
        let runtime_errors_for_callback = runtime_errors.clone();
        device.on_uncaptured_error(Arc::new(move |error| {
            runtime_errors_for_callback
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(error.to_string());
        }));
        let device_lost = Arc::new(std::sync::Mutex::new(None));
        let device_lost_for_callback = device_lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            *device_lost_for_callback
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(format!("{reason:?}: {message}"));
        });

        let (simd_lane_count, simd_mul_add_fused) = crate::filter::active_gpu_simd_profile();

        let chroma_into_luma_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chroma_into_luma shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chroma_into_luma.wgsl").into()),
        });

        let luma_into_chroma_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("luma_into_chroma shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    filter_arithmetic_source(),
                    include_str!("shaders/luma_into_chroma.wgsl")
                )
                .into(),
            ),
        });

        let chroma_delay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chroma_delay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chroma_delay.wgsl").into()),
        });

        let filter_plane_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filter_plane shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/filter_plane.wgsl").into()),
        });

        let simplex_src = include_str!("shaders/simplex.wgsl");

        let chroma_loss_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chroma vertical blend"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chroma_loss_blend.wgsl").into()),
        });

        let image_layout_entries: [wgpu::BindGroupLayoutEntry; IMAGE_STORAGE_BINDINGS as usize] =
            std::array::from_fn(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("copy bind group layout"),
                entries: &image_layout_entries,
            });

        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("params bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let filter_coeffs_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("filter coeffs bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let effect_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("effect pipeline layout"),
                bind_group_layouts: &[&copy_bind_group_layout, &params_bind_group_layout],
                push_constant_ranges: &[],
            });

        let filter_plane_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("filter_plane pipeline layout"),
                bind_group_layouts: &[
                    &copy_bind_group_layout,
                    &params_bind_group_layout,
                    &filter_coeffs_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let row_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("row control data"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let row_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("row effect layout"),
            bind_group_layouts: &[
                &copy_bind_group_layout,
                &params_bind_group_layout,
                &row_layout,
            ],
            push_constant_ranges: &[],
        });
        let row_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reference row effects"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    simplex_src,
                    include_str!("shaders/row_effects.wgsl")
                )
                .into(),
            ),
        });
        let row_pipelines = ["noise", "shift_y", "shift_all", "phase", "loss", "snow"]
            .into_iter()
            .map(|entry| {
                (
                    entry,
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: Some(&row_pipeline_layout),
                        module: &row_shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
                )
            })
            .collect();

        let chroma_into_luma_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma_into_luma pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &chroma_into_luma_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let luma_into_chroma_box_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("luma_into_chroma box pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &luma_into_chroma_shader,
                entry_point: Some("demodulate_box"),
                compilation_options: Default::default(),
                cache: None,
            });

        let make_demodulation = |entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&effect_pipeline_layout),
                module: &luma_into_chroma_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let luma_into_chroma_notch_pipeline = make_demodulation("demodulate_notch");
        let luma_into_chroma_one_line_pipeline = make_demodulation("demodulate_one_line_comb");
        let luma_into_chroma_two_line_pipeline = make_demodulation("demodulate_two_line_comb");
        let luma_box_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("input luma box"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/luma_box.wgsl").into()),
        });
        let luma_box_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("input luma box"),
            layout: Some(&effect_pipeline_layout),
            module: &luma_box_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let phase_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chroma phase"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/chroma_phase.wgsl").into()),
        });
        let chroma_phase_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma phase"),
                layout: Some(&effect_pipeline_layout),
                module: &phase_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let chroma_delay_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma_delay pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &chroma_delay_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let filter_plane_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("filter_plane pipeline"),
                layout: Some(&filter_plane_pipeline_layout),
                module: &filter_plane_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let chroma_vert_blend_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("chroma_vert_blend pipeline"),
                layout: Some(&effect_pipeline_layout),
                module: &chroma_loss_shader,
                entry_point: Some("chroma_vert_blend"),
                compilation_options: Default::default(),
                cache: None,
            });

        let params = ShaderParams {
            width: 0,
            frame_num: 0,
            seed: 0,
            noise_idx: 0,

            noise_frequency: 0.0,
            noise_intensity: 0.0,
            noise_detail: 0,
            snow_anisotropy: 0.0,

            phase_shift: 0,
            phase_offset: 0,
            filter_mode: 0,
            chroma_delay_horizontal: 0.0,

            chroma_delay_vertical: 0,
            horizontal_scale: 1.0,
            vertical_scale: 1.0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
        };

        let mut params_ring_buffer = Vec::new();
        for _ in 0..32 {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shader params buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("params bind group"),
                layout: &params_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            params_ring_buffer.push((buffer, bind_group));
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            copy_bind_group_layout,
            chroma_into_luma_pipeline,
            luma_into_chroma_box_pipeline,
            luma_into_chroma_notch_pipeline,
            luma_into_chroma_one_line_pipeline,
            luma_into_chroma_two_line_pipeline,
            luma_box_pipeline,
            chroma_phase_pipeline,
            chroma_delay_pipeline,
            filter_plane_pipeline,
            chroma_vert_blend_pipeline,
            filter_coeffs_bind_group_layout,
            params_ring_buffer,
            row_layout,
            row_pipelines,
            data_cache: Default::default(),
            data_index: Default::default(),
            filter_cache: Default::default(),
            adapter_info: adapter.get_info(),
            adapter_limits,
            requested_limits,
            simd_lane_count,
            simd_mul_add_fused,
            runtime_errors,
            device_lost,
            dispatch_stages: Default::default(),
            cpu_control_stages: Default::default(),
            pending_error: Default::default(),
            submitted_effects: 0,
        })
    }

    /// Number of effect command buffers submitted by this backend instance.
    /// Readback-only submissions are deliberately excluded.
    pub fn submitted_effects(&self) -> u64 {
        self.submitted_effects
    }

    pub fn frame_capacity_requirements(
        &self,
        width: usize,
        rows: usize,
    ) -> Result<FrameCapacityRequirements, WgpuBackendError> {
        let requirements = FrameCapacityRequirements::new(width, rows)?;
        requirements.validate(&self.device.limits())?;
        Ok(requirements)
    }

    fn record_dispatch(&self, stage: &'static str) {
        self.dispatch_stages.borrow_mut().push(stage);
    }

    fn record_cpu_control(&self, stage: &'static str) {
        self.cpu_control_stages.borrow_mut().push(stage);
    }

    pub fn execution_evidence(&self) -> (Vec<&'static str>, Vec<&'static str>) {
        (
            self.dispatch_stages.borrow().clone(),
            self.cpu_control_stages.borrow().clone(),
        )
    }

    pub fn begin_execution(&self) {
        self.runtime_errors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.pending_error.borrow_mut().take();
    }

    pub fn current_device_loss(&self) -> Option<String> {
        self.device_lost
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn take_pending_error(&self) -> Option<WgpuBackendError> {
        self.pending_error.borrow_mut().take()
    }

    fn dispatch_filter_plane<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &WgpuFrame,
        params_bind_group: &'a wgpu::BindGroup,
        tf: &crate::filter::TransferFunction,
        first_sample: bool,
        delay: usize,
        plane_idx: u32,
    ) {
        self.record_dispatch("filter_plane");
        let (num, den, z_initial) = tf.to_gpu_coeffs(if first_sample { 1.0 } else { 0.0 });
        let filter_coeffs = FilterCoeffs {
            num,
            den,
            z_initial,
            delay: delay as u32,
            filter_len: tf.len() as u32,
            plane_idx,
            // Bit 0 selects FirstSample initial conditions. Bit 1 selects the
            // fused arithmetic used by the active upstream SIMD backend.
            initial_condition_mode: u32::from(first_sample)
                | (u32::from(
                    self.simd_mul_add_fused
                        && self.simd_lane_count != 0
                        && (2..=4).contains(&tf.len()),
                ) << 1),
        };

        let key: [u32; 16] = bytemuck::cast(filter_coeffs);
        let mut cache = self.filter_cache.borrow_mut();
        if !cache.contains_key(&key) && cache.len() >= 256 {
            cache.clear();
        }
        let coeffs_bind_group = cache.entry(key).or_insert_with(|| {
            let coeffs_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("filter coeffs buffer"),
                    contents: bytemuck::cast_slice(&[filter_coeffs]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("filter coeffs bind group"),
                layout: &self.filter_coeffs_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: coeffs_buffer.as_entire_binding(),
                }],
            })
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("filter_plane pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.filter_plane_pipeline);
        cpass.set_bind_group(0, &frame.main_bind_group, &[]);
        cpass.set_bind_group(1, params_bind_group, &[]);
        cpass.set_bind_group(2, &*coeffs_bind_group, &[]);

        let rows = frame.height as u32;
        cpass.dispatch_workgroups(rows.div_ceil(64), 1, 1);
    }

    fn dispatch_data(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &WgpuFrame,
        params: &wgpu::BindGroup,
        main: &wgpu::BindGroup,
        entry: &'static str,
        data: &[u8],
    ) {
        self.dispatch_shared_data(encoder, frame, params, main, &[entry], data);
    }

    fn dispatch_shared_data(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &WgpuFrame,
        params: &wgpu::BindGroup,
        main: &wgpu::BindGroup,
        entries: &[&'static str],
        data: &[u8],
    ) {
        if data.is_empty() {
            return;
        }
        if self.pending_error.borrow().is_some() {
            return;
        }
        let index = self.data_index.get();
        self.data_index.set(index + 1);
        let mut cache = self.data_cache.borrow_mut();
        let size = data.len() as u64;
        let limits = self.device.limits();
        let capacity = match storage_buffer_capacity(size, &limits) {
            Ok(capacity) => capacity,
            Err(error) => {
                *self.pending_error.borrow_mut() = Some(error);
                return;
            }
        };
        if index >= cache.len() || cache[index].2 < size {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: entries.first().copied(),
                size: capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: entries.first().copied(),
                layout: &self.row_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            let slot = (buffer, group, capacity);
            if index >= cache.len() {
                cache.push(slot);
            } else {
                cache[index] = slot;
            }
        }
        let (buffer, group, _) = &cache[index];
        self.queue.write_buffer(buffer, 0, data);
        for &entry in entries {
            self.record_dispatch(entry);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(entry),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.row_pipelines[entry]);
            pass.set_bind_group(0, main, &[]);
            pass.set_bind_group(1, params, &[]);
            pass.set_bind_group(2, group, &[]);
            if entry == "noise" {
                pass.dispatch_workgroups(1, frame.height as u32, 1);
            } else {
                pass.dispatch_workgroups(
                    (frame.width as u32).div_ceil(16),
                    (frame.height as u32).div_ceil(16),
                    1,
                );
            }
        }
    }

    fn dispatch_chroma_lowpass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &WgpuFrame,
        params: &wgpu::BindGroup,
        mode: crate::settings::standard::ChromaLowpass,
        filter_type: crate::settings::standard::FilterType,
        scale: f32,
    ) {
        use crate::{
            ntsc::{NTSC_RATE, make_lowpass_for_type},
            settings::standard::ChromaLowpass,
        };
        let cuts = match mode {
            ChromaLowpass::None => return,
            ChromaLowpass::Light => [(2_600_000.0, 1), (2_600_000.0, 1)],
            ChromaLowpass::Full => [(1_300_000.0, 2), (600_000.0, 4)],
        };
        for (idx, (cutoff, delay)) in cuts.into_iter().enumerate() {
            let filter = make_lowpass_for_type(cutoff, NTSC_RATE * scale, filter_type);
            self.dispatch_filter_plane(
                encoder,
                frame,
                params,
                &filter,
                false,
                delay,
                idx as u32 + 1,
            );
        }
    }

    fn get_params_bind_group<'a>(
        &'a self,
        params: &ShaderParams,
        ring_idx: &mut usize,
    ) -> &'a wgpu::BindGroup {
        let idx = *ring_idx;
        assert!(
            idx < self.params_ring_buffer.len(),
            "uniform slots exhausted within one submission"
        );
        *ring_idx += 1;
        let (buffer, bind_group) = &self.params_ring_buffer[idx];
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&[*params]));
        bind_group
    }
}

impl GpuBackend for WgpuBackend {
    type Frame = WgpuFrame;

    fn upload_frame(&mut self, src: &YiqView) -> Self::Frame {
        let requirements = self
            .frame_capacity_requirements(src.dimensions.0, src.num_rows())
            .expect("WGPU frame capacity must be validated before allocation");
        let size = requirements.plane_bytes;

        let create_buffer = |label: &str, data: &[f32]| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&data[..data.len()]));
            buffer
        };

        let y_buffer = create_buffer("y_buffer", src.y);
        let i_buffer = create_buffer("i_buffer", src.i);
        let q_buffer = create_buffer("q_buffer", src.q);
        let scratch_buffers = std::array::from_fn(|plane| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["scratch_y", "scratch_i", "scratch_q"][plane]),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let staging_buffers = std::array::from_fn(|plane| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["readback_y", "readback_i", "readback_q"][plane]),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let make_bind_group = |label, images: [&wgpu::Buffer; 3]| {
            let resources = [
                images[0],
                images[1],
                images[2],
                &scratch_buffers[0],
                &scratch_buffers[1],
                &scratch_buffers[2],
            ];
            let entries: [wgpu::BindGroupEntry; IMAGE_STORAGE_BINDINGS as usize] =
                std::array::from_fn(|binding| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: resources[binding].as_entire_binding(),
                });
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.copy_bind_group_layout,
                entries: &entries,
            })
        };
        let main_bind_group = make_bind_group("main bind group", [&y_buffer, &i_buffer, &q_buffer]);
        let i_pass_bind_group =
            make_bind_group("i pass bind group", [&i_buffer, &y_buffer, &q_buffer]);
        let q_pass_bind_group =
            make_bind_group("q pass bind group", [&q_buffer, &y_buffer, &i_buffer]);
        let chroma_loss_bind_group =
            make_bind_group("chroma loss bind group", [&i_buffer, &q_buffer, &y_buffer]);

        WgpuFrame {
            y_buffer,
            i_buffer,
            q_buffer,
            scratch_buffers,
            staging_buffers,
            main_bind_group,
            i_pass_bind_group,
            q_pass_bind_group,
            chroma_loss_bind_group,
            width: src.dimensions.0,
            height: src.num_rows(),
            full_height: src.dimensions.1,
            device: self.device.clone(),
            queue: self.queue.clone(),
            runtime_errors: self.runtime_errors.clone(),
            device_lost: self.device_lost.clone(),
        }
    }

    fn apply_effect(
        &mut self,
        effect: &NtscEffect,
        frame: &mut Self::Frame,
        frame_num: usize,
        scale_factor: [f32; 2],
    ) {
        self.data_index.set(0);
        self.dispatch_stages.borrow_mut().clear();
        self.cpu_control_stages.borrow_mut().clear();
        use super::prepare;
        let size = (frame.width * frame.height * std::mem::size_of::<f32>()) as u64;
        let mut ring_idx = 0;
        let [horizontal_scale, vertical_scale] =
            crate::ntsc::effective_scale_factors(effect, frame.full_height, scale_factor);
        let mut params = ShaderParams {
            width: frame.width as u32,
            frame_num: frame_num as u32,
            seed: effect.random_seed as u32,
            noise_idx: 0,

            noise_frequency: 0.0,
            noise_intensity: 0.0,
            noise_detail: 0,
            snow_anisotropy: 0.0,

            phase_shift: effect.video_scanline_phase_shift as u32,
            phase_offset: effect.video_scanline_phase_shift_offset,
            filter_mode: effect.chroma_demodulation as u32,
            chroma_delay_horizontal: effect.chroma_delay_horizontal,

            chroma_delay_vertical: effect.chroma_delay_vertical,
            horizontal_scale,
            vertical_scale,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute encoder"),
            });

        let main_bind_group = &frame.main_bind_group;

        let num_pixels = frame.width * frame.height;
        let row_workgroups = (frame.height as u32).div_ceil(64);
        let pixel_workgroups = (
            (frame.width as u32).div_ceil(16),
            (frame.height as u32).div_ceil(16),
        );

        use crate::{
            ntsc::{NTSC_RATE, make_lowpass, make_lowpass_for_type, make_notch_filter},
            settings::standard::{ChromaDemodulationFilter, FilterType, LumaLowpass},
        };
        let base_params = self.get_params_bind_group(&params, &mut ring_idx);
        match effect.input_luma_filter {
            LumaLowpass::None => {}
            LumaLowpass::Notch => self.dispatch_filter_plane(
                &mut encoder,
                frame,
                base_params,
                &make_notch_filter(0.5, 2.0),
                true,
                0,
                0,
            ),
            LumaLowpass::Box => {
                self.record_dispatch("luma_box");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.luma_box_pipeline);
                pass.set_bind_group(0, main_bind_group, &[]);
                pass.set_bind_group(1, base_params, &[]);
                pass.dispatch_workgroups(row_workgroups, 1, 1);
            }
        }
        self.dispatch_chroma_lowpass(
            &mut encoder,
            frame,
            base_params,
            effect.chroma_lowpass_in,
            effect.filter_type,
            params.horizontal_scale,
        );

        {
            self.record_dispatch("chroma_into_luma");
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_into_luma pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_into_luma_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, base_params, &[]);
            cpass.dispatch_workgroups(pixel_workgroups.0, pixel_workgroups.1, 1);
        }

        if effect.composite_sharpening != 0.0 {
            let filter = make_lowpass(
                (315000000.0 / 88.0 / 2.0) * params.horizontal_scale,
                NTSC_RATE * params.horizontal_scale,
            )
            .with_scale(-effect.composite_sharpening);
            self.dispatch_filter_plane(&mut encoder, frame, base_params, &filter, false, 0, 0);
        }

        let sx = params.horizontal_scale;
        let sy = params.vertical_scale;
        if let Some(noise) = &effect.composite_noise {
            self.record_cpu_control("composite_noise_rows");
            let rows = prepare::noise(
                effect.random_seed,
                frame_num,
                noise_seeds::VIDEO_COMPOSITE,
                frame.width,
                frame.height,
                sx,
                noise,
            );
            self.dispatch_data(
                &mut encoder,
                frame,
                base_params,
                main_bind_group,
                "noise",
                bytemuck::cast_slice(&rows),
            );
        }
        if effect.snow_intensity > 0.0 && sx > 0.0 {
            self.record_cpu_control("snow_events_and_values");
            let events = prepare::snow(
                effect.random_seed,
                frame_num,
                frame.width,
                frame.height,
                effect.snow_intensity * 0.01,
                effect.snow_anisotropy,
                sx,
            );
            self.dispatch_data(
                &mut encoder,
                frame,
                base_params,
                main_bind_group,
                "snow",
                bytemuck::cast_slice(&events),
            );
        }
        if let Some(head) = &effect.head_switching {
            self.record_cpu_control("head_switching_rows");
            let rows = prepare::head(
                effect.random_seed,
                frame_num,
                frame.width,
                frame.height,
                sx,
                sy,
                head,
            );
            encoder.copy_buffer_to_buffer(&frame.y_buffer, 0, &frame.scratch_buffers[0], 0, size);
            self.dispatch_data(
                &mut encoder,
                frame,
                base_params,
                main_bind_group,
                "shift_y",
                bytemuck::cast_slice(&rows),
            );
        }
        if let Some(tracking) = &effect.tracking_noise {
            self.record_cpu_control("tracking_displacement_noise_and_snow");
            let (rows, events) = prepare::tracking(
                effect.random_seed,
                frame_num,
                frame.width,
                frame.height,
                sx,
                sy,
                tracking,
            );
            encoder.copy_buffer_to_buffer(&frame.y_buffer, 0, &frame.scratch_buffers[0], 0, size);
            self.dispatch_shared_data(
                &mut encoder,
                frame,
                base_params,
                main_bind_group,
                &["shift_y", "noise"],
                bytemuck::cast_slice(&rows),
            );
            self.dispatch_data(
                &mut encoder,
                frame,
                base_params,
                main_bind_group,
                "snow",
                bytemuck::cast_slice(&events),
            );
        }

        let size = (num_pixels * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(&frame.y_buffer, 0, &frame.scratch_buffers[0], 0, size);

        params.phase_shift = effect.video_scanline_phase_shift as u32;
        params.phase_offset = effect.video_scanline_phase_shift_offset;
        params.filter_mode = effect.chroma_demodulation as u32;

        let demodulation = if frame.height == 1
            && matches!(
                effect.chroma_demodulation,
                ChromaDemodulationFilter::OneLineComb | ChromaDemodulationFilter::TwoLineComb
            ) {
            ChromaDemodulationFilter::Notch
        } else {
            effect.chroma_demodulation
        };
        let demod_params = base_params;
        if demodulation == ChromaDemodulationFilter::Notch {
            self.dispatch_filter_plane(
                &mut encoder,
                frame,
                demod_params,
                &make_notch_filter(0.5, 2.0),
                false,
                0,
                0,
            );
        }
        {
            self.record_dispatch("luma_into_chroma");
            let pipeline = match demodulation {
                ChromaDemodulationFilter::Box => &self.luma_into_chroma_box_pipeline,
                ChromaDemodulationFilter::Notch => &self.luma_into_chroma_notch_pipeline,
                ChromaDemodulationFilter::OneLineComb => &self.luma_into_chroma_one_line_pipeline,
                ChromaDemodulationFilter::TwoLineComb => &self.luma_into_chroma_two_line_pipeline,
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, main_bind_group, &[]);
            pass.set_bind_group(1, demod_params, &[]);
            pass.dispatch_workgroups(pixel_workgroups.0, pixel_workgroups.1, 1);
        }
        if effect.luma_smear > 0.0 {
            let filter = make_lowpass(
                f32::exp2(-4.0 * effect.luma_smear) * 0.25,
                params.horizontal_scale,
            );
            self.dispatch_filter_plane(&mut encoder, frame, demod_params, &filter, false, 0, 0);
        }
        if let Some(ringing) = &effect.ringing {
            let filter = make_notch_filter(
                (ringing.frequency / params.horizontal_scale).clamp(0.0, 1.0),
                ringing.power,
            )
            .with_scale(ringing.intensity);
            self.dispatch_filter_plane(&mut encoder, frame, demod_params, &filter, true, 1, 0);
        }

        if let Some(noise) = &effect.luma_noise {
            self.record_cpu_control("luma_noise_rows");
            let rows = prepare::noise(
                effect.random_seed,
                frame_num,
                noise_seeds::VIDEO_LUMA,
                frame.width,
                frame.height,
                sx,
                noise,
            );
            self.dispatch_data(
                &mut encoder,
                frame,
                base_params,
                main_bind_group,
                "noise",
                bytemuck::cast_slice(&rows),
            );
        }
        if let Some(noise) = &effect.chroma_noise {
            for (tag, group) in [
                (noise_seeds::VIDEO_CHROMA_I, &frame.i_pass_bind_group),
                (noise_seeds::VIDEO_CHROMA_Q, &frame.q_pass_bind_group),
            ] {
                self.record_cpu_control(if tag == noise_seeds::VIDEO_CHROMA_I {
                    "chroma_i_noise_rows"
                } else {
                    "chroma_q_noise_rows"
                });
                let rows = prepare::noise(
                    effect.random_seed,
                    frame_num,
                    tag,
                    frame.width,
                    frame.height,
                    sx,
                    noise,
                );
                self.dispatch_data(
                    &mut encoder,
                    frame,
                    base_params,
                    group,
                    "noise",
                    bytemuck::cast_slice(&rows),
                );
            }
        }

        if effect.chroma_phase_error > 0.0 {
            self.record_dispatch("chroma_phase_error");
            params.noise_frequency = effect.chroma_phase_error;
            params.noise_intensity = 0.0;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.chroma_phase_pipeline);
            pass.set_bind_group(0, main_bind_group, &[]);
            pass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            pass.dispatch_workgroups(pixel_workgroups.0, pixel_workgroups.1, 1);
        }

        if effect.chroma_phase_noise_intensity > 0.0 {
            self.record_cpu_control("chroma_phase_noise_rows");
            let rows = prepare::phase(
                effect.random_seed,
                frame_num,
                frame.height,
                effect.chroma_phase_noise_intensity,
            );
            self.dispatch_data(
                &mut encoder,
                frame,
                base_params,
                main_bind_group,
                "phase",
                bytemuck::cast_slice(&rows),
            );
        }

        if effect.chroma_delay_horizontal != 0.0 || effect.chroma_delay_vertical != 0 {
            self.record_dispatch("chroma_delay");
            params.chroma_delay_horizontal =
                effect.chroma_delay_horizontal * params.horizontal_scale;
            params.chroma_delay_vertical =
                (effect.chroma_delay_vertical as f32 * params.vertical_scale).round() as i32;
            encoder.copy_buffer_to_buffer(&frame.i_buffer, 0, &frame.scratch_buffers[1], 0, size);
            encoder.copy_buffer_to_buffer(&frame.q_buffer, 0, &frame.scratch_buffers[2], 0, size);

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_delay pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_delay_pipeline);
            cpass.set_bind_group(0, main_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(pixel_workgroups.0, pixel_workgroups.1, 1);
        }

        if let Some(vhs_settings) = &effect.vhs_settings {
            if let Some(wave) = &vhs_settings.edge_wave
                && wave.intensity > 0.0
            {
                self.record_cpu_control("vhs_edge_wave_displacement");
                let rows = prepare::wave(effect.random_seed, frame_num, frame.height, sx, sy, wave);
                for (plane, buffer) in [&frame.y_buffer, &frame.i_buffer, &frame.q_buffer]
                    .into_iter()
                    .enumerate()
                {
                    encoder.copy_buffer_to_buffer(
                        buffer,
                        0,
                        &frame.scratch_buffers[plane],
                        0,
                        size,
                    );
                }
                self.dispatch_data(
                    &mut encoder,
                    frame,
                    base_params,
                    main_bind_group,
                    "shift_all",
                    bytemuck::cast_slice(&rows),
                );
            }

            if let Some(tape) = vhs_settings.tape_speed.filter_params() {
                let rate = NTSC_RATE * params.horizontal_scale;
                let delay = (tape.chroma_delay as f32 * params.horizontal_scale).round() as usize;
                let luma = make_lowpass_for_type(tape.luma_cut, rate, effect.filter_type);
                let chroma = make_lowpass_for_type(tape.chroma_cut, rate, effect.filter_type);
                self.dispatch_filter_plane(&mut encoder, frame, base_params, &luma, false, 0, 0);
                for plane in 1..=2 {
                    self.dispatch_filter_plane(
                        &mut encoder,
                        frame,
                        base_params,
                        &chroma,
                        false,
                        delay,
                        plane,
                    );
                }
                self.dispatch_filter_plane(
                    &mut encoder,
                    frame,
                    base_params,
                    &make_lowpass(tape.luma_cut, rate).with_scale(-1.6),
                    false,
                    0,
                    0,
                );
            }

            if vhs_settings.chroma_loss > 0.0 {
                self.record_cpu_control("chroma_loss_rows");
                let rows = prepare::loss(
                    effect.random_seed,
                    frame_num,
                    frame.height,
                    vhs_settings.chroma_loss,
                );
                self.dispatch_data(
                    &mut encoder,
                    frame,
                    base_params,
                    main_bind_group,
                    "loss",
                    bytemuck::cast_slice(&rows),
                );
            }
        }

        if let Some(vhs) = &effect.vhs_settings
            && let (Some(sharpen), Some(tape)) = (&vhs.sharpen, vhs.tape_speed.filter_params())
        {
            let extra = match effect.filter_type {
                FilterType::ConstantK => 4.0,
                FilterType::Butterworth => 1.0,
            };
            let filter = make_lowpass_for_type(
                tape.luma_cut * extra * sharpen.frequency,
                NTSC_RATE * params.horizontal_scale,
                effect.filter_type,
            )
            .with_scale(-sharpen.intensity * 2.0 * sharpen.frequency);
            self.dispatch_filter_plane(&mut encoder, frame, base_params, &filter, false, 0, 0);
        }

        if effect.chroma_vert_blend && frame.full_height >= 2 {
            self.record_dispatch("chroma_vert_blend");
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_vert_blend pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.chroma_vert_blend_pipeline);
            cpass.set_bind_group(0, &frame.chroma_loss_bind_group, &[]);
            cpass.set_bind_group(1, self.get_params_bind_group(&params, &mut ring_idx), &[]);
            cpass.dispatch_workgroups(frame.width.div_ceil(64) as u32, 1, 1);
        }

        self.dispatch_chroma_lowpass(
            &mut encoder,
            frame,
            base_params,
            effect.chroma_lowpass_out,
            effect.filter_type,
            params.horizontal_scale,
        );

        if self.pending_error.borrow().is_some() {
            return;
        }
        self.queue.submit(Some(encoder.finish()));
        self.submitted_effects += 1;
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn split_plane_requirements_cover_large_progressive_and_field_frames() {
        let progressive = FrameCapacityRequirements::new(7680, 4320).unwrap();
        assert_eq!(progressive.plane_bytes, 132_710_400);
        assert_eq!(progressive.combined_plane_bytes, 398_131_200);
        progressive.validate(&wgpu::Limits::default()).unwrap();

        let field = FrameCapacityRequirements::new(7680, 2160).unwrap();
        assert_eq!(field.plane_bytes, 66_355_200);
        assert_eq!(field.combined_plane_bytes, 199_065_600);
        field.validate(&wgpu::Limits::default()).unwrap();
    }

    #[test]
    fn capacity_rejects_each_relevant_limit_before_allocation() {
        let too_wide = FrameCapacityRequirements::new(8192, 4320).unwrap();
        assert_eq!(too_wide.plane_bytes, 141_557_760);
        assert!(matches!(
            too_wide.validate(&wgpu::Limits::default()),
            Err(WgpuBackendError::UnsupportedCapacity {
                resource: "image plane storage binding",
                ..
            })
        ));

        let requirement = FrameCapacityRequirements::new(64, 64).unwrap();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = u32::MAX;
        limits.max_buffer_size = requirement.plane_bytes - 1;
        assert!(matches!(
            requirement.validate(&limits),
            Err(WgpuBackendError::UnsupportedCapacity {
                resource: "image plane buffer",
                ..
            })
        ));

        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = IMAGE_STORAGE_BINDINGS - 1;
        assert!(matches!(
            requirement.validate(&limits),
            Err(WgpuBackendError::UnsupportedCapacity {
                resource: "storage buffers per shader stage",
                ..
            })
        ));

        let mut limits = wgpu::Limits::default();
        limits.max_compute_workgroups_per_dimension = 1;
        assert!(matches!(
            requirement.validate(&limits),
            Err(WgpuBackendError::UnsupportedCapacity {
                resource: "horizontal compute workgroups",
                ..
            })
        ));
    }

    #[test]
    fn capacity_arithmetic_is_checked() {
        assert!(matches!(
            FrameCapacityRequirements::new(0, 1),
            Err(WgpuBackendError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            FrameCapacityRequirements::new(usize::MAX, usize::MAX),
            Err(WgpuBackendError::SizeOverflow { .. })
        ));
    }

    #[test]
    fn control_buffer_capacity_is_checked_and_bounded() {
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 1_000;
        limits.max_buffer_size = 2_000;

        assert_eq!(storage_buffer_capacity(513, &limits).unwrap(), 1_000);
        assert_eq!(storage_buffer_capacity(1_000, &limits).unwrap(), 1_000);
        assert!(matches!(
            storage_buffer_capacity(1_001, &limits),
            Err(WgpuBackendError::UnsupportedCapacity {
                resource: "control-data storage binding",
                required: 1_001,
                supported: 1_000,
            })
        ));

        limits.max_storage_buffer_binding_size = 4_096;
        limits.max_buffer_size = 1_500;
        assert_eq!(storage_buffer_capacity(1_024, &limits).unwrap(), 1_024);
        assert_eq!(storage_buffer_capacity(1_025, &limits).unwrap(), 1_500);
    }
}
