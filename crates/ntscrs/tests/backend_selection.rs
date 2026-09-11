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
fn explicit_cpu_reports_cpu_and_unimplemented_cuda_is_an_error() {
    let effect = NtscEffect::default();
    let dimensions = (8, 8);
    let input = vec![0.2; YiqView::buf_length_for(dimensions, YiqField::Both)];
    let mut expected = input.clone();
    effect.apply_effect_to_yiq(
        &mut YiqView::from_parts(&mut expected, dimensions, YiqField::Both),
        3,
        [1.0, 1.0],
    );
    let mut actual = input.clone();
    let execution = apply_effect_to_yiq_with_backend_preference(
        &effect,
        &mut YiqView::from_parts(&mut actual, dimensions, YiqField::Both),
        3,
        [1.0, 1.0],
        BackendPreference::Cpu,
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(execution.requested, ntsc_rs::gpu::BackendType::Cpu);
    assert_eq!(execution.actual, ntsc_rs::gpu::BackendType::Cpu);
    assert_eq!(execution.cpu_image_effect_invocations, 1);

    let mut cuda_actual = input.clone();
    let error = apply_effect_to_yiq_with_backend_preference(
        &effect,
        &mut YiqView::from_parts(&mut cuda_actual, dimensions, YiqField::Both),
        3,
        [1.0, 1.0],
        BackendPreference::Cuda,
    )
    .unwrap_err();
    assert_eq!(error.requested, ntsc_rs::gpu::BackendType::Cuda);
    assert_eq!(error.kind, ntsc_rs::gpu::BackendFailureKind::Unavailable);
    assert_eq!(
        cuda_actual, input,
        "an unavailable explicit backend must not process on CPU"
    );
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
            let info = runner.wgpu_adapter_info().unwrap();
            assert!(!info.name.is_empty());
            let (adapter, requested, device) = runner.wgpu_limit_snapshot().unwrap();
            assert_eq!(
                requested.max_storage_buffer_binding_size,
                adapter.max_storage_buffer_binding_size
            );
            assert_eq!(requested.max_buffer_size, adapter.max_buffer_size);
            assert_eq!(
                device.max_storage_buffer_binding_size,
                requested.max_storage_buffer_binding_size
            );
            assert_eq!(device.max_buffer_size, requested.max_buffer_size);
            eprintln!(
                "adapter={} backend={:?} device_type={:?} adapter_storage_binding={} requested_storage_binding={} device_storage_binding={} adapter_buffer={} requested_buffer={} device_buffer={}",
                info.name,
                info.backend,
                info.device_type,
                adapter.max_storage_buffer_binding_size,
                requested.max_storage_buffer_binding_size,
                device.max_storage_buffer_binding_size,
                adapter.max_buffer_size,
                requested.max_buffer_size,
                device.max_buffer_size
            );
        }
        BackendType::Cpu => {
            assert!(runner.fallback_reason().is_some());
            assert!(runner.wgpu_adapter_info().is_none());
        }
        BackendType::Auto | BackendType::Cuda => {
            panic!("backend request must resolve to a concrete implementation")
        }
    }
}

#[cfg(not(feature = "gpu-wgpu"))]
#[test]
fn explicit_wgpu_without_feature_is_an_error_and_leaves_pixels_unchanged() {
    let effect = NtscEffect::default();
    let dimensions = (8, 8);
    let mut data = vec![0.2; YiqView::buf_length_for(dimensions, YiqField::Both)];
    let original = data.clone();
    let error = apply_effect_to_yiq_with_backend_preference(
        &effect,
        &mut YiqView::from_parts(&mut data, dimensions, YiqField::Both),
        3,
        [1.0, 1.0],
        BackendPreference::Wgpu,
    )
    .unwrap_err();
    assert_eq!(error.kind, ntsc_rs::gpu::BackendFailureKind::Unavailable);
    assert_eq!(data, original);
}
