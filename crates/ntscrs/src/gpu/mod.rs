#[cfg(feature = "gpu-wgpu")]
pub mod block_filter;
#[cfg(feature = "gpu-wgpu")]
pub mod profiling;
use crate::{settings::standard::NtscEffect, yiq_fielding::YiqView};

#[cfg(feature = "gpu-wgpu")]
mod prepare;
pub mod runner;
#[cfg(feature = "gpu-wgpu")]
pub mod wgpu_backend;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// Always use the CPU.
    Cpu,
    /// Use WGPU for GPU acceleration.
    Wgpu,
    /// Reserved for the unimplemented CUDA backend.
    Cuda,
    /// Automatically select the best available backend (WGPU if supported, otherwise CPU).
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFailureKind {
    Unavailable,
    Initialization,
    UnsupportedCapacity,
    Runtime,
    Readback,
    DeviceLost,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError {
    pub requested: BackendType,
    pub kind: BackendFailureKind,
    pub message: String,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "requested {:?} backend failed ({:?}): {}",
            self.requested, self.kind, self.message
        )
    }
}

impl std::error::Error for BackendError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendExecution {
    pub requested: BackendType,
    pub actual: BackendType,
    pub fallback_reason: Option<BackendError>,
    /// Number of calls into the CPU image-effect entry point during this execution.
    pub cpu_image_effect_invocations: u64,
    /// WGPU entry points encoded for this execution. This is regression instrumentation,
    /// not cryptographic attestation.
    pub dispatched_stages: Vec<&'static str>,
    /// CPU-generated control signals uploaded for shader application.
    pub cpu_control_stages: Vec<&'static str>,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::Auto
    }
}

/// Abstract representation of a frame or buffer that lives on the GPU.
/// It must be able to hold the Y, I, Q, and scratch planes, and allow for downloading
/// the processed results back into a `YiqView`.
pub trait GpuFrame {
    /// Download the GPU-resident frame data back into the provided `YiqView`.
    /// This is a blocking operation.
    fn download(&self, dst: &mut YiqView);
}

/// A common interface for all backends (CPU, WGPU, and potentially CUDA in the future).
pub trait GpuBackend {
    type Frame: GpuFrame;

    /// Upload a `YiqView` from CPU to the GPU.
    fn upload_frame(&mut self, src: &YiqView) -> Self::Frame;

    /// Process the frame using the provided settings.
    fn apply_effect(
        &mut self,
        effect: &NtscEffect,
        frame: &mut Self::Frame,
        frame_num: usize,
        scale_factor: [f32; 2],
    );
}
