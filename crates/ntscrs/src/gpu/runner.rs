#[cfg(feature = "gpu-wgpu")]
use crate::yiq_fielding::YiqField;
use crate::{
    gpu::{BackendType, GpuBackend, GpuFrame},
    settings::standard::NtscEffect,
    yiq_fielding::YiqView,
};

/// The CPU reference implementation wrapper, adhering to the GpuBackend interface
/// so that the pipeline runner can transparently call it.
pub struct CpuBackend;

pub struct CpuFrame<'a> {
    pub yiq: YiqView<'a>,
}

impl<'a> GpuFrame for CpuFrame<'a> {
    fn download(&self, _dst: &mut YiqView) {
        // Since we process in-place on the CPU, a download is essentially a no-op
        // if they refer to the same buffers. However, to match the semantics where
        // `CpuFrame` might be an ephemeral wrapper, if it's different data, we'd copy.
        // For simplicity in the runner, we handle it by not copying at all if we are on CPU.
    }
}

impl GpuBackend for CpuBackend {
    type Frame = CpuFrame<'static>;

    fn upload_frame(&mut self, _src: &YiqView) -> Self::Frame {
        // Note: The CPU backend modifies in place. The runner handles this specially.
        unimplemented!("CPU backend handles frames in place");
    }

    fn apply_effect(
        &mut self,
        effect: &NtscEffect,
        frame: &mut Self::Frame,
        frame_num: usize,
        scale_factor: [f32; 2],
    ) {
        effect.apply_effect_to_yiq(&mut frame.yiq, frame_num, scale_factor);
    }
}

pub struct NtscEffectRunner {
    backend_type: BackendType,
    last_backend: BackendType,
    fallback_reason: Option<&'static str>,
    #[cfg(feature = "gpu-wgpu")]
    frames: [Option<crate::gpu::wgpu_backend::WgpuFrame>; 2],
    #[cfg(feature = "gpu-wgpu")]
    wgpu_backend: Option<crate::gpu::wgpu_backend::WgpuBackend>,
}

impl NtscEffectRunner {
    pub fn new(requested_backend: BackendType) -> Self {
        #[allow(unused_mut)]
        let mut actual_backend = BackendType::Cpu;

        #[cfg(feature = "gpu-wgpu")]
        let mut wgpu_backend = None;

        match requested_backend {
            BackendType::Cpu => {}
            #[cfg(feature = "gpu-wgpu")]
            BackendType::Wgpu | BackendType::Auto => {
                if let Some(backend) = crate::gpu::wgpu_backend::WgpuBackend::new() {
                    // A software Vulkan adapter executes shaders on the CPU and must not
                    // be advertised as automatic hardware acceleration. Explicit Wgpu
                    // remains available for CI shader validation and diagnostics.
                    if requested_backend != BackendType::Auto
                        || backend.adapter_info.device_type != wgpu::DeviceType::Cpu
                    {
                        eprintln!(
                            "ntsc-rs: using GPU adapter {} ({:?})",
                            backend.adapter_info.name, backend.adapter_info.backend
                        );
                        wgpu_backend = Some(backend);
                        actual_backend = BackendType::Wgpu;
                    }
                } else {
                    println!("ntsc-rs: Failed to initialize WGPU backend, falling back to CPU.");
                }
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            BackendType::Auto => {}
        }

        Self {
            backend_type: actual_backend,
            last_backend: BackendType::Cpu,
            fallback_reason: None,
            #[cfg(feature = "gpu-wgpu")]
            frames: [None, None],
            #[cfg(feature = "gpu-wgpu")]
            wgpu_backend,
        }
    }

    pub fn active_backend(&self) -> BackendType {
        self.backend_type
    }

    /// Backend that actually rendered the most recent frame, including per-effect fallbacks.
    pub fn last_backend(&self) -> BackendType {
        self.last_backend
    }

    pub fn fallback_reason(&self) -> Option<&'static str> {
        self.fallback_reason
    }

    pub fn apply_effect(
        &mut self,
        src: &mut YiqView,
        effect: &NtscEffect,
        frame_num: usize,
        scale_factor: [f32; 2],
    ) {
        self.last_backend = BackendType::Cpu;
        self.fallback_reason = None;
        match self.backend_type {
            BackendType::Cpu => {
                effect.apply_effect_to_yiq(src, frame_num, scale_factor);
            }
            #[cfg(feature = "gpu-wgpu")]
            BackendType::Wgpu => {
                let backend = self.wgpu_backend.as_mut().unwrap();
                let dimensions = src.dimensions;
                if src.y.is_empty() {
                    return;
                }
                let (mut first, mut second, first_num, second_num) = match src.field {
                    YiqField::InterleavedUpper => {
                        let (mut upper, mut lower) =
                            src.split_at_row(YiqField::Upper.num_actual_image_rows(dimensions.1));
                        if let Some(view) = &mut upper {
                            view.field = YiqField::Upper;
                        }
                        if let Some(view) = &mut lower {
                            view.field = YiqField::Lower;
                        }
                        (upper, lower, frame_num * 2, frame_num * 2 + 1)
                    }
                    YiqField::InterleavedLower => {
                        let (mut lower, mut upper) =
                            src.split_at_row(YiqField::Lower.num_actual_image_rows(dimensions.1));
                        if let Some(view) = &mut lower {
                            view.field = YiqField::Lower;
                        }
                        if let Some(view) = &mut upper {
                            view.field = YiqField::Upper;
                        }
                        (lower, upper, frame_num * 2, frame_num * 2 + 1)
                    }
                    _ => (
                        Some(YiqView {
                            y: src.y,
                            i: src.i,
                            q: src.q,
                            scratch: src.scratch,
                            dimensions: src.dimensions,
                            field: src.field,
                        }),
                        None,
                        frame_num,
                        frame_num,
                    ),
                };
                let mut pending = [None, None];
                // Submit both fields before waiting for either readback. Each field has
                // its own buffers; odd heights and the reference field timebase are preserved.
                for (slot, view, number) in [
                    (0, first.as_ref(), first_num),
                    (1, second.as_ref(), second_num),
                ] {
                    if let Some(view) = view {
                        let frame = &mut self.frames[slot];
                        if frame.as_ref().is_none_or(|frame| {
                            (frame.width, frame.height) != (view.dimensions.0, view.num_rows())
                        }) {
                            *frame = Some(backend.upload_frame(view));
                        } else {
                            backend.upload_into(view, frame.as_mut().unwrap());
                        }
                        let frame = frame.as_mut().unwrap();
                        backend.apply_effect(effect, frame, number, scale_factor);
                        pending[slot] = Some(frame.enqueue_download());
                    }
                }
                for (slot, view) in [(0, first.as_mut()), (1, second.as_mut())] {
                    if let Some(view) = view {
                        self.frames[slot]
                            .as_ref()
                            .unwrap()
                            .finish_download(view, pending[slot].take().unwrap());
                    }
                }
                self.last_backend = BackendType::Wgpu;
            }
            BackendType::Auto => unreachable!("Auto should have resolved to a concrete backend"),
        }
    }
}
