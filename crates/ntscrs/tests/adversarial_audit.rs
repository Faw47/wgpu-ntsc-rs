//! Acceptance checks derived from pinned upstream 88f2df9.
use ntsc_rs::{
    NtscEffectFullSettings,
    settings::{SettingsList, easy::EasyModeFullSettings, standard::*},
    yiq_fielding::YiqField,
};

#[test]
fn alternating_mapping_matches_pinned_upstream() {
    for n in [0, 1, 8, 9] {
        assert_eq!(
            UseField::Alternating.to_yiq_field(n),
            if n % 2 == 0 {
                YiqField::Lower
            } else {
                YiqField::Upper
            }
        );
    }
}

#[test]
fn legacy_missing_noise_values_match_pinned_upstream() {
    let full = SettingsList::<NtscEffectFullSettings>::new()
        .from_json(r#"{"version":1}"#)
        .unwrap();
    assert_eq!(full.composite_noise.settings.intensity, 0.05);
    assert_eq!(full.chroma_noise.settings.intensity, 0.1);
    assert_eq!(full.luma_noise.settings.frequency, 0.5);
    assert_eq!(full.luma_noise.settings.intensity, 0.01);
    assert_eq!(full.luma_noise.settings.detail, 1);
}

#[test]
fn partial_legacy_noise_imports_match_pinned_upstream() {
    let list = SettingsList::<NtscEffectFullSettings>::new();

    let composite = list
        .from_json(r#"{"version":1,"composite_noise":true,"composite_noise_frequency":0.75}"#)
        .unwrap();
    assert!(composite.composite_noise.enabled);
    assert_eq!(composite.composite_noise.settings.frequency, 0.75);
    assert_eq!(composite.composite_noise.settings.intensity, 0.05);
    assert_eq!(composite.composite_noise.settings.detail, 1);

    let chroma = list
        .from_json(r#"{"version":1,"chroma_noise":true,"chroma_noise_detail":4}"#)
        .unwrap();
    assert!(chroma.chroma_noise.enabled);
    assert_eq!(chroma.chroma_noise.settings.frequency, 0.05);
    assert_eq!(chroma.chroma_noise.settings.intensity, 0.1);
    assert_eq!(chroma.chroma_noise.settings.detail, 4);

    let luma = list
        .from_json(r#"{"version":1,"luma_noise":true}"#)
        .unwrap();
    assert!(luma.luma_noise.enabled);
    assert_eq!(luma.luma_noise.settings.frequency, 0.5);
    assert_eq!(luma.luma_noise.settings.intensity, 0.01);
    assert_eq!(luma.luma_noise.settings.detail, 1);

    let serialized = list.to_json_string(&luma).unwrap();
    assert_eq!(list.from_json(&serialized).unwrap(), luma);
}

#[test]
fn settings_surface_and_easy_lowering_match_pinned_upstream() {
    let standard = SettingsList::<NtscEffectFullSettings>::new();
    let easy = SettingsList::<EasyModeFullSettings>::new();
    assert_eq!(standard.all_descriptors().count(), 62);
    assert_eq!(easy.all_descriptors().count(), 26);

    let easy_settings = EasyModeFullSettings::default();
    let lowered = NtscEffectFullSettings::from(&easy_settings);
    assert_eq!(lowered.input_luma_filter, LumaLowpass::Notch);
    assert_eq!(lowered.chroma_lowpass_in, ChromaLowpass::None);
    assert_eq!(lowered.video_scanline_phase_shift, PhaseShift::Degrees180);
    assert_eq!(lowered.video_scanline_phase_shift_offset, 0);
    assert!(!lowered.composite_noise.enabled);
    assert_eq!(lowered.composite_sharpening, 0.25);
    assert_eq!(lowered.snow_intensity, 0.0005);
    assert_eq!(lowered.chroma_phase_noise_intensity, 0.005);
    assert_eq!(lowered.chroma_lowpass_out, ChromaLowpass::Full);
    assert!(lowered.head_switching.enabled);
    assert_eq!(lowered.head_switching.settings.height, 8);
    assert_eq!(lowered.head_switching.settings.offset, 2);
    assert_eq!(lowered.head_switching.settings.horiz_shift, 48.0);
    assert!(lowered.tracking_noise.enabled);
    assert_eq!(lowered.tracking_noise.settings.wave_intensity, 2.5);
    assert_eq!(lowered.tracking_noise.settings.snow_intensity, 0.25);
    assert_eq!(lowered.tracking_noise.settings.noise_intensity, 0.25);
    assert!(lowered.ringing.enabled);
    assert_eq!(lowered.ringing.settings.frequency, 0.4);
    assert_eq!(lowered.ringing.settings.power, 2.0);
    assert_eq!(lowered.ringing.settings.intensity, 0.5);

    let serialized = easy.to_json_string(&easy_settings).unwrap();
    assert_eq!(easy.from_json_generic(&serialized).unwrap(), easy_settings);
}

#[cfg(feature = "gpu-wgpu")]
mod numerical {
    use super::*;
    use ntsc_rs::{
        NtscEffect,
        gpu::{GpuBackend, GpuFrame, wgpu_backend::WgpuBackend},
        yiq_fielding::YiqView,
    };
    fn clean() -> NtscEffect {
        let mut e = NtscEffect::default();
        e.input_luma_filter = LumaLowpass::None;
        e.chroma_lowpass_in = ChromaLowpass::None;
        e.chroma_lowpass_out = ChromaLowpass::None;
        e.chroma_demodulation = ChromaDemodulationFilter::Box;
        e.luma_smear = 0.0;
        e.composite_sharpening = 0.0;
        e.head_switching = None;
        e.tracking_noise = None;
        e.ringing = None;
        e.composite_noise = None;
        e.luma_noise = None;
        e.chroma_noise = None;
        e.snow_intensity = 0.0;
        e.chroma_phase_noise_intensity = 0.0;
        e.chroma_phase_error = 0.0;
        e.vhs_settings = None;
        e.chroma_vert_blend = false;
        e.scale = None;
        e
    }
    fn diff(
        gpu: &mut WgpuBackend,
        e: &NtscEffect,
        w: usize,
        h: usize,
        scale: [f32; 2],
        label: &str,
    ) -> f32 {
        let mut source = vec![0.0; YiqView::buf_length_for((w, h), YiqField::Both)];
        {
            let v = YiqView::from_parts(&mut source, (w, h), YiqField::Both);
            for idx in 0..w * h {
                v.y[idx] = ((idx * 13 + idx / w * 7) % 256) as f32 / 255.0;
                v.i[idx] = ((idx * 5 + 11) % 97) as f32 / 200.0 - 0.24;
                v.q[idx] = ((idx * 3 + 17) % 89) as f32 / 200.0 - 0.22;
            }
        }
        let mut expected = source.clone();
        let mut cv = YiqView::from_parts(&mut expected, (w, h), YiqField::Both);
        e.apply_effect_to_yiq(&mut cv, 13, scale);
        let mut gv = YiqView::from_parts(&mut source, (w, h), YiqField::Both);
        let mut frame = gpu.upload_frame(&gv);
        let before = gpu.submitted_effects();
        gpu.apply_effect(e, &mut frame, 13, scale);
        assert_eq!(gpu.submitted_effects(), before + 1);
        frame.download(&mut gv);
        let mut error = 0.0f32;
        for (a, b) in [(cv.y, gv.y), (cv.i, gv.i), (cv.q, gv.q)] {
            for (a, b) in a.iter().zip(b) {
                assert!(a.is_finite() && b.is_finite(), "{label}: nonfinite");
                error = error.max((*a - *b).abs());
            }
        }
        eprintln!("AUDIT {label} {w}x{h} scale={scale:?} max={error}");
        error
    }
    #[test]
    #[ignore = "requires a compute adapter, including software Vulkan"]
    fn noise_extremes_and_sd_hd_dimensions() {
        let mut gpu = WgpuBackend::new().expect("adapter required");
        eprintln!("ADAPTER {:?}", gpu.adapter_info);
        let mut failures = Vec::new();
        for w in [257, 720, 1920, 3840] {
            for sx in [0.125, 1.0, 8.0] {
                for seed in [0, -47] {
                    let mut e = clean();
                    e.random_seed = seed;
                    e.luma_noise = Some(FbmNoiseSettings {
                        frequency: 1.0,
                        intensity: 1.0,
                        detail: 5,
                    });
                    let err = diff(&mut gpu, &e, w, 3, [sx, 1.0], "luma-octaves5");
                    if err > 0.002 {
                        failures.push((w, sx, seed, err));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "pinned noise parity failures: {failures:?}"
        );
    }

    #[test]
    #[ignore = "requires a compute adapter, including software Vulkan"]
    fn every_shared_additive_noise_path_matches_cpu_at_wide_scales() {
        let mut gpu = WgpuBackend::new().expect("adapter required");
        let mut failures = Vec::new();
        for width in [1920, 3840] {
            for scale in [0.125, 1.0] {
                for seed in [0, -47] {
                    for path in ["composite", "luma", "chroma", "tracking"] {
                        let mut effect = clean();
                        effect.random_seed = seed;
                        let noise = FbmNoiseSettings {
                            frequency: 1.0,
                            intensity: 1.0,
                            detail: 5,
                        };
                        match path {
                            "composite" => effect.composite_noise = Some(noise),
                            "luma" => effect.luma_noise = Some(noise),
                            "chroma" => effect.chroma_noise = Some(noise),
                            "tracking" => {
                                effect.tracking_noise = Some(TrackingNoiseSettings {
                                    height: 3,
                                    wave_intensity: 0.0,
                                    snow_intensity: 0.0,
                                    snow_anisotropy: 0.0,
                                    noise_intensity: 1.0,
                                })
                            }
                            _ => unreachable!(),
                        }
                        let error = diff(&mut gpu, &effect, width, 3, [scale, 1.0], path);
                        if error > 0.002 {
                            failures.push((path, width, scale, seed, error));
                        }
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "shared additive-noise parity failures: {failures:?}"
        );
    }
    #[test]
    #[ignore = "requires a compute adapter, including software Vulkan"]
    fn phase_modes_and_filter_extremes() {
        let mut gpu = WgpuBackend::new().expect("adapter required");
        let mut failures = Vec::new();
        for mode in [
            PhaseShift::Degrees0,
            PhaseShift::Degrees90,
            PhaseShift::Degrees180,
            PhaseShift::Degrees270,
        ] {
            for offset in [-3, 1, 3] {
                let mut e = clean();
                e.video_scanline_phase_shift = mode;
                e.video_scanline_phase_shift_offset = offset;
                let err = diff(&mut gpu, &e, 65, 3, [1.0, 1.0], "phase-mode");
                if err > 0.002 {
                    failures.push(format!("phase {mode:?}/{offset}: {err}"));
                }
            }
        }
        for typ in [FilterType::ConstantK, FilterType::Butterworth] {
            for sx in [0.125, 8.0] {
                let mut e = clean();
                e.filter_type = typ;
                e.chroma_lowpass_in = ChromaLowpass::Full;
                e.chroma_lowpass_out = ChromaLowpass::Full;
                e.ringing = Some(RingingSettings {
                    frequency: 1.0,
                    power: 10.0,
                    intensity: 10.0,
                });
                let err = diff(&mut gpu, &e, 720, 3, [sx, 1.0], "filter-extreme");
                if err > 0.002 {
                    failures.push(format!("filter {typ:?}/{sx}: {err}"));
                }
            }
        }
        for sx in [0.125, 8.0] {
            let mut e = clean();
            e.head_switching = Some(HeadSwitchingSettings {
                height: 3,
                offset: 0,
                horiz_shift: 20.0,
                mid_line: Some(HeadSwitchingMidLineSettings::default()),
            });
            let err = diff(&mut gpu, &e, 3840, 3, [sx, 1.0], "head-shift-extreme");
            if err > 0.002 {
                failures.push(format!("head shift/{sx}: {err}"));
            }
        }
        assert!(failures.is_empty(), "parity failures: {failures:?}");
    }
}
