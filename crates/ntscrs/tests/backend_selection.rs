use ntsc_rs::{
    BackendPreference, NtscEffect, apply_effect_to_yiq_with_backend_preference,
    yiq_fielding::{YiqField, YiqView},
};
use std::str::FromStr;

#[test]
fn parses_actual_backend_preferences() {
    for (name, preference) in [
        (" CPU ", BackendPreference::Cpu),
        ("WGPU", BackendPreference::Wgpu),
        ("cuda", BackendPreference::Cuda),
        ("auto", BackendPreference::Auto),
    ] {
        assert_eq!(BackendPreference::from_str(name), Ok(preference));
    }
    assert!(BackendPreference::from_str("not-a-backend").is_err());
}

#[test]
fn explicit_cpu_and_unimplemented_cuda_preserve_cpu_output() {
    let effect = NtscEffect::default();
    let dimensions = (8, 8);
    let input = vec![0.2; YiqView::buf_length_for(dimensions, YiqField::Both)];
    let mut expected = input.clone();
    effect.apply_effect_to_yiq(
        &mut YiqView::from_parts(&mut expected, dimensions, YiqField::Both),
        3,
        [1.0, 1.0],
    );
    for preference in [BackendPreference::Cpu, BackendPreference::Cuda] {
        let mut actual = input.clone();
        apply_effect_to_yiq_with_backend_preference(
            &effect,
            &mut YiqView::from_parts(&mut actual, dimensions, YiqField::Both),
            3,
            [1.0, 1.0],
            preference,
        );
        assert_eq!(actual, expected);
    }
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn explicit_wgpu_selection_reports_adapter_or_fallback_reason() {
    use ntsc_rs::gpu::{BackendType, runner::NtscEffectRunner};

    let runner = NtscEffectRunner::new(BackendType::Wgpu);
    assert_eq!(runner.requested_backend(), BackendType::Wgpu);
    match runner.active_backend() {
        BackendType::Wgpu => {
            assert!(runner.fallback_reason().is_none());
            assert!(runner.wgpu_adapter_info().is_some());
        }
        BackendType::Cpu => {
            assert!(runner.fallback_reason().is_some());
            assert!(runner.wgpu_adapter_info().is_none());
        }
        BackendType::Auto => panic!("Auto must resolve to a concrete backend"),
    }
}
