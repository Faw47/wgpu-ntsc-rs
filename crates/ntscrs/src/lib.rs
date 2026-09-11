mod backend;
mod filter;
pub mod gpu;
mod noise;
mod ntsc;
#[cfg(feature = "gpu-wgpu")]
pub(crate) use ntsc::noise_seeds;
mod random;
pub mod settings;
mod shift;
mod thread_pool;
pub mod yiq_fielding;

use std::str::FromStr;

pub use settings::standard::{NtscEffect, NtscEffectFullSettings};
use yiq_fielding::YiqView;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

impl BackendPreference {
    pub fn from_env_var(var_name: &str) -> Option<Self> {
        let value = std::env::var(var_name).ok()?;
        Self::from_str(value.trim()).ok()
    }
}

impl FromStr for BackendPreference {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "wgpu" => Ok(Self::Wgpu),
            "cuda" => Ok(Self::Cuda),
            _ => Err(()),
        }
    }
}

pub fn apply_effect_to_yiq_with_backend_preference(
    effect: &NtscEffect,
    yiq: &mut YiqView,
    frame_num: usize,
    scale_factor: [f32; 2],
    backend_preference: BackendPreference,
) -> Result<gpu::BackendExecution, gpu::BackendError> {
    // A GStreamer worker processes many frames. Keep its GPU device, compiled pipelines,
    // and frame buffers alive instead of initializing the entire backend for every frame.
    #[cfg(feature = "gpu-wgpu")]
    if matches!(
        backend_preference,
        BackendPreference::Auto | BackendPreference::Wgpu
    ) {
        thread_local! {
            static REQUESTED: std::cell::Cell<BackendPreference> = const { std::cell::Cell::new(BackendPreference::Auto) };
            // Construct wgpu before registering this TLS destructor. Its debug lock
            // tracing also uses TLS and must remain alive while GPU resources drop.
            static RUNNER: std::cell::RefCell<(BackendPreference, gpu::runner::NtscEffectRunner)> = {
                let preference = REQUESTED.get();
                std::cell::RefCell::new((preference, make_runner(preference)))
            };
        }
        fn make_runner(preference: BackendPreference) -> gpu::runner::NtscEffectRunner {
            let requested = if preference == BackendPreference::Auto {
                gpu::BackendType::Auto
            } else {
                gpu::BackendType::Wgpu
            };
            gpu::runner::NtscEffectRunner::new(requested)
        }
        REQUESTED.set(backend_preference);
        return RUNNER.with(|runner| {
            let mut cached = runner.borrow_mut();
            if cached.0 != backend_preference {
                *cached = (backend_preference, make_runner(backend_preference));
            }
            let (_, runner) = &mut *cached;
            runner.apply_effect(yiq, effect, frame_num, scale_factor)
        });
    }
    let requested = match backend_preference {
        BackendPreference::Auto => gpu::BackendType::Auto,
        BackendPreference::Cpu => gpu::BackendType::Cpu,
        BackendPreference::Wgpu => gpu::BackendType::Wgpu,
        BackendPreference::Cuda => gpu::BackendType::Cuda,
    };
    gpu::runner::NtscEffectRunner::new(requested).apply_effect(yiq, effect, frame_num, scale_factor)
}
