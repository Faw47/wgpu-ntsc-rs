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
