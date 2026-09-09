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
    /// Use WGPU for GPU acceleration. Falls back to CPU if WGPU initialization fails.
    #[cfg(feature = "gpu-wgpu")]
    Wgpu,
    /// Automatically select the best available backend (WGPU if supported, otherwise CPU).
    Auto,
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
