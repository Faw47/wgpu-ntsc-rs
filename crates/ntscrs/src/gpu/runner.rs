#[cfg(feature = "gpu-wgpu")]
use crate::yiq_fielding::YiqField;
use crate::{
    gpu::{BackendError, BackendExecution, BackendFailureKind, BackendType, GpuBackend, GpuFrame},
    settings::standard::NtscEffect,
    yiq_fielding::YiqView,
};

#[cfg(feature = "gpu-wgpu")]
fn classify_wgpu_error(
    requested: BackendType,
    error: crate::gpu::wgpu_backend::WgpuBackendError,
) -> BackendError {
    use crate::gpu::wgpu_backend::WgpuBackendError;
    let kind = match &error {
        WgpuBackendError::Initialization(_) => BackendFailureKind::Initialization,
        WgpuBackendError::InvalidDimensions { .. }
        | WgpuBackendError::SizeOverflow { .. }
        | WgpuBackendError::UnsupportedCapacity { .. } => BackendFailureKind::UnsupportedCapacity,
        WgpuBackendError::Runtime(_) => BackendFailureKind::Runtime,
        WgpuBackendError::Readback(_) => BackendFailureKind::Readback,
        WgpuBackendError::DeviceLost(_) => BackendFailureKind::DeviceLost,
    };
    BackendError {
        requested,
        kind,
        message: error.to_string(),
    }
}

pub struct CpuBackend;

pub struct CpuFrame<'a> {
    pub yiq: YiqView<'a>,
}

impl<'a> GpuFrame for CpuFrame<'a> {
    fn download(&self, _dst: &mut YiqView) {}
}

impl GpuBackend for CpuBackend {
    type Frame = CpuFrame<'static>;

    fn upload_frame(&mut self, _src: &YiqView) -> Self::Frame {
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
    requested_backend: BackendType,
    backend_type: BackendType,
    last_backend: BackendType,
    fallback_reason: Option<BackendError>,
    #[cfg(feature = "gpu-wgpu")]
    frames: [Option<crate::gpu::wgpu_backend::WgpuFrame>; 2],
    #[cfg(feature = "gpu-wgpu")]
    wgpu_backend: Option<crate::gpu::wgpu_backend::WgpuBackend>,
}

impl NtscEffectRunner {
    pub fn new(requested_backend: BackendType) -> Self {
        #[allow(unused_mut)]
        let mut actual_backend = BackendType::Cpu;
        #[allow(unused_mut)]
        let mut fallback_reason = None;

        #[cfg(feature = "gpu-wgpu")]
        let mut wgpu_backend = None;

        match requested_backend {
            BackendType::Cpu => {}
            #[cfg(feature = "gpu-wgpu")]
            BackendType::Wgpu | BackendType::Auto => {
                match crate::gpu::wgpu_backend::WgpuBackend::try_new() {
                    Ok(backend) => {
                        if requested_backend != BackendType::Auto
                            || backend.adapter_info.device_type != wgpu::DeviceType::Cpu
                        {
                            eprintln!(
                                "ntsc-rs: using GPU adapter {} ({:?})",
                                backend.adapter_info.name, backend.adapter_info.backend
                            );
                            wgpu_backend = Some(backend);
                            actual_backend = BackendType::Wgpu;
                        } else {
                            fallback_reason = Some(BackendError {
                                requested: requested_backend,
                                kind: BackendFailureKind::Unavailable,
                                message: "automatic selection rejected a CPU WGPU adapter"
                                    .to_owned(),
                            });
                        }
                    }
                    Err(error) => {
                        fallback_reason = Some(BackendError {
                            requested: requested_backend,
                            kind: BackendFailureKind::Initialization,
                            message: error.to_string(),
                        });
                    }
                }
            }
            BackendType::Cuda => {
                fallback_reason = Some(BackendError {
                    requested: requested_backend,
                    kind: BackendFailureKind::Unavailable,
                    message: "CUDA backend is not implemented".to_owned(),
                });
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            BackendType::Wgpu | BackendType::Auto => {
                fallback_reason = Some(BackendError {
                    requested: requested_backend,
                    kind: BackendFailureKind::Unavailable,
                    message: "ntsc-rs was built without gpu-wgpu support".to_owned(),
                });
            }
        }

        Self {
            requested_backend,
            backend_type: actual_backend,
            last_backend: BackendType::Cpu,
            fallback_reason,
            #[cfg(feature = "gpu-wgpu")]
            frames: [None, None],
            #[cfg(feature = "gpu-wgpu")]
            wgpu_backend,
        }
    }

    pub fn active_backend(&self) -> BackendType {
        self.backend_type
    }

    pub fn requested_backend(&self) -> BackendType {
        self.requested_backend
    }

    pub fn last_backend(&self) -> BackendType {
        self.last_backend
    }

    pub fn fallback_reason(&self) -> Option<&BackendError> {
        self.fallback_reason.as_ref()
    }

    #[cfg(feature = "gpu-wgpu")]
    pub fn wgpu_adapter_info(&self) -> Option<&wgpu::AdapterInfo> {
        self.wgpu_backend
            .as_ref()
            .map(|backend| &backend.adapter_info)
    }

    #[cfg(feature = "gpu-wgpu")]
    pub fn wgpu_submitted_effects(&self) -> u64 {
        self.wgpu_backend
            .as_ref()
            .map_or(0, |backend| backend.submitted_effects())
    }

    #[cfg(feature = "gpu-wgpu")]
    pub fn wgpu_limit_snapshot(&self) -> Option<(wgpu::Limits, wgpu::Limits, wgpu::Limits)> {
        self.wgpu_backend.as_ref().map(|backend| {
            (
                backend.adapter_limits.clone(),
                backend.requested_limits.clone(),
                backend.device.limits(),
            )
        })
    }

    #[cfg(all(test, feature = "gpu-wgpu"))]
    pub(crate) fn destroy_wgpu_device_for_test(&self) {
        let backend = self
            .wgpu_backend
            .as_ref()
            .expect("test requires an active WGPU backend");
        *backend
            .device_lost
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some("test-injected device destruction".to_owned());
        backend.device.destroy();
    }

    pub fn apply_effect(
        &mut self,
        src: &mut YiqView,
        effect: &NtscEffect,
        frame_num: usize,
        scale_factor: [f32; 2],
    ) -> Result<BackendExecution, BackendError> {
        self.last_backend = BackendType::Cpu;
        match self.backend_type {
            BackendType::Cpu => {
                if matches!(
                    self.requested_backend,
                    BackendType::Wgpu | BackendType::Cuda
                ) {
                    return Err(self
                        .fallback_reason
                        .clone()
                        .unwrap_or_else(|| BackendError {
                            requested: self.requested_backend,
                            kind: BackendFailureKind::Unavailable,
                            message: "requested backend is unavailable".to_owned(),
                        }));
                }
                #[cfg(feature = "gpu-wgpu")]
                let cpu_before = crate::ntsc::cpu_image_effect_invocations();
                effect.apply_effect_to_yiq(src, frame_num, scale_factor);
                #[cfg(feature = "gpu-wgpu")]
                let cpu_image_effect_invocations =
                    crate::ntsc::cpu_image_effect_invocations() - cpu_before;
                #[cfg(not(feature = "gpu-wgpu"))]
                let cpu_image_effect_invocations = 1;
                Ok(BackendExecution {
                    requested: self.requested_backend,
                    actual: BackendType::Cpu,
                    fallback_reason: self.fallback_reason.clone(),
                    cpu_image_effect_invocations,
                    dispatched_stages: Vec::new(),
                    cpu_control_stages: Vec::new(),
                })
            }
            #[cfg(feature = "gpu-wgpu")]
            BackendType::Wgpu => {
                self.fallback_reason = None;
                let backend = self.wgpu_backend.as_mut().unwrap();
                let dimensions = src.dimensions;
                if src.y.is_empty() {
                    self.last_backend = BackendType::Wgpu;
                    return Ok(BackendExecution {
                        requested: self.requested_backend,
                        actual: BackendType::Wgpu,
                        fallback_reason: None,
                        cpu_image_effect_invocations: 0,
                        dispatched_stages: Vec::new(),
                        cpu_control_stages: Vec::new(),
                    });
                }

                let row_counts = match src.field {
                    YiqField::InterleavedUpper | YiqField::InterleavedLower => [
                        YiqField::Upper.num_actual_image_rows(dimensions.1),
                        YiqField::Lower.num_actual_image_rows(dimensions.1),
                    ],
                    _ => [src.num_rows(), 0],
                };
                for rows in row_counts.into_iter().filter(|rows| *rows != 0) {
                    if let Err(error) = backend.frame_capacity_requirements(dimensions.0, rows) {
                        let error = classify_wgpu_error(self.requested_backend, error);
                        self.fallback_reason = Some(error.clone());
                        if self.requested_backend == BackendType::Auto {
                            let cpu_before = crate::ntsc::cpu_image_effect_invocations();
                            effect.apply_effect_to_yiq(src, frame_num, scale_factor);
                            return Ok(BackendExecution {
                                requested: self.requested_backend,
                                actual: BackendType::Cpu,
                                fallback_reason: Some(error),
                                cpu_image_effect_invocations:
                                    crate::ntsc::cpu_image_effect_invocations() - cpu_before,
                                dispatched_stages: Vec::new(),
                                cpu_control_stages: Vec::new(),
                            });
                        }
                        return Err(error);
                    }
                }
                let cpu_before = crate::ntsc::cpu_image_effect_invocations();
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
                let mut dispatched_stages = Vec::new();
                let mut cpu_control_stages = Vec::new();
                backend.begin_execution();
                if let Some(message) = backend.current_device_loss() {
                    let error = classify_wgpu_error(
                        self.requested_backend,
                        crate::gpu::wgpu_backend::WgpuBackendError::DeviceLost(message),
                    );
                    self.fallback_reason = Some(error.clone());
                    return Err(error);
                }
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
                        if let Some(error) = backend.take_pending_error() {
                            let error = classify_wgpu_error(self.requested_backend, error);
                            self.fallback_reason = Some(error.clone());
                            return Err(error);
                        }
                        let (field_dispatches, field_controls) = backend.execution_evidence();
                        dispatched_stages.extend(field_dispatches);
                        cpu_control_stages.extend(field_controls);
                        pending[slot] = Some(frame.enqueue_download());
                    }
                }
                let mut readback_error = None;
                for (slot, view) in [(0, first.as_mut()), (1, second.as_mut())] {
                    if let Some(view) = view {
                        if let Err(error) = self.frames[slot]
                            .as_ref()
                            .unwrap()
                            .try_finish_download(view, pending[slot].take().unwrap())
                        {
                            readback_error.get_or_insert(error);
                        }
                    }
                }
                if let Some(error) = readback_error {
                    let error = classify_wgpu_error(self.requested_backend, error);
                    self.fallback_reason = Some(error.clone());
                    return Err(error);
                }
                self.last_backend = BackendType::Wgpu;
                Ok(BackendExecution {
                    requested: self.requested_backend,
                    actual: BackendType::Wgpu,
                    fallback_reason: None,
                    cpu_image_effect_invocations: crate::ntsc::cpu_image_effect_invocations()
                        - cpu_before,
                    dispatched_stages,
                    cpu_control_stages,
                })
            }
            BackendType::Auto | BackendType::Cuda => {
                unreachable!("backend requests must resolve before execution")
            }
            #[cfg(not(feature = "gpu-wgpu"))]
            BackendType::Wgpu => unreachable!("unavailable WGPU must resolve to CPU"),
        }
    }
}
