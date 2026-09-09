#![cfg(feature = "gpu-wgpu")]

use ntsc_rs::{
    NtscEffect,
    gpu::{GpuBackend, GpuFrame, wgpu_backend::WgpuBackend},
    settings::standard::*,
    yiq_fielding::{YiqField, YiqView},
};

fn clean_effect() -> NtscEffect {
    let mut effect = NtscEffect::default();
    effect.use_field = UseField::Both;
    effect.input_luma_filter = LumaLowpass::None;
    effect.chroma_lowpass_in = ChromaLowpass::None;
    effect.chroma_lowpass_out = ChromaLowpass::None;
    effect.chroma_demodulation = ChromaDemodulationFilter::Box;
    effect.luma_smear = 0.0;
    effect.composite_sharpening = 0.0;
    effect.head_switching = None;
    effect.tracking_noise = None;
    effect.ringing = None;
    effect.composite_noise = None;
    effect.luma_noise = None;
    effect.chroma_noise = None;
    effect.snow_intensity = 0.0;
    effect.chroma_phase_noise_intensity = 0.0;
    effect.vhs_settings = None;
    effect.chroma_vert_blend = false;
    effect
}

fn compare(
    backend: &mut WgpuBackend,
    effect: &NtscEffect,
    width: usize,
    height: usize,
    frame_num: usize,
    scale: [f32; 2],
    label: &str,
) {
    let mut input = vec![0.0; YiqView::buf_length_for((width, height), YiqField::Both)];
    {
        let view = YiqView::from_parts(&mut input, (width, height), YiqField::Both);
        for idx in 0..width * height {
            view.y[idx] = ((idx * 13 + idx / width * 7) % 256) as f32 / 255.0;
            view.i[idx] = ((idx * 5 + 11) % 97) as f32 / 200.0 - 0.24;
            view.q[idx] = ((idx * 3 + 17) % 89) as f32 / 200.0 - 0.22;
        }
    }
    let mut cpu = input.clone();
    let mut cpu_view = YiqView::from_parts(&mut cpu, (width, height), YiqField::Both);
    effect.apply_effect_to_yiq(&mut cpu_view, frame_num, scale);
    let mut gpu_view = YiqView::from_parts(&mut input, (width, height), YiqField::Both);
    let mut frame = backend.upload_frame(&gpu_view);
    backend.apply_effect(effect, &mut frame, frame_num, scale);
    frame.download(&mut gpu_view);
    for (plane, expected, actual) in [
        ("Y", cpu_view.y, gpu_view.y),
        ("I", cpu_view.i, gpu_view.i),
        ("Q", cpu_view.q, gpu_view.q),
    ] {
        assert_eq!(expected.len(), actual.len());
        let mut max_error = 0.0f32;
        for (idx, (&a, &b)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                a.is_finite() && b.is_finite(),
                "{label}/{plane}[{idx}] nonfinite: {a}, {b}"
            );
            max_error = max_error.max((a - b).abs());
        }
        assert!(
            max_error <= 2e-3,
            "{label} {width}x{height} frame={frame_num} scale={scale:?} {plane}: max error {max_error}"
        );
    }
}

// Explicitly ignored on hosts without a compute adapter. CI must invoke --ignored;
// unlike the old test, this fails if GPU initialization fails and never runs CPU emulation.
#[test]
#[ignore = "requires a Vulkan, Metal, or DX12 compute adapter"]
fn deterministic_effect_matrix_matches_cpu() {
    let mut gpu =
        WgpuBackend::new().expect("a real wgpu adapter is required; CPU fallback is forbidden");
    let mut cases = vec![("base", clean_effect())];
    for (label, mode) in [
        ("box", ChromaDemodulationFilter::Box),
        ("notch", ChromaDemodulationFilter::Notch),
        ("one-line", ChromaDemodulationFilter::OneLineComb),
        ("two-line", ChromaDemodulationFilter::TwoLineComb),
    ] {
        let mut effect = clean_effect();
        effect.chroma_demodulation = mode;
        cases.push((label, effect));
    }
    for (label, mode) in [
        ("luma-box", LumaLowpass::Box),
        ("luma-notch", LumaLowpass::Notch),
    ] {
        let mut effect = clean_effect();
        effect.input_luma_filter = mode;
        cases.push((label, effect));
    }
    for (label, mode) in [
        ("chroma-light", ChromaLowpass::Light),
        ("chroma-full", ChromaLowpass::Full),
    ] {
        for filter in [FilterType::Butterworth, FilterType::ConstantK] {
            let mut effect = clean_effect();
            effect.filter_type = filter;
            effect.chroma_lowpass_in = mode;
            effect.chroma_lowpass_out = mode;
            cases.push((label, effect));
        }
    }
    let mut effect = clean_effect();
    effect.composite_sharpening = 1.0;
    cases.push(("composite-sharpen", effect));
    let mut effect = clean_effect();
    effect.luma_smear = 0.5;
    cases.push(("smear", effect));
    let mut effect = clean_effect();
    effect.ringing = Some(RingingSettings::default());
    cases.push(("ringing", effect));
    let mut effect = clean_effect();
    effect.chroma_phase_error = 0.125;
    cases.push(("phase", effect));
    let mut effect = clean_effect();
    effect.chroma_vert_blend = true;
    cases.push(("vertical-blend", effect));
    for shift in [-3.5, 2.25] {
        let mut effect = clean_effect();
        effect.chroma_delay_horizontal = shift;
        effect.chroma_delay_vertical = if shift < 0.0 { -2 } else { 2 };
        cases.push(("delay", effect));
    }
    for speed in [VHSTapeSpeed::SP, VHSTapeSpeed::LP, VHSTapeSpeed::EP] {
        for filter in [FilterType::Butterworth, FilterType::ConstantK] {
            let mut effect = clean_effect();
            effect.filter_type = filter;
            effect.vhs_settings = Some(VHSSettings {
                tape_speed: speed,
                chroma_loss: 0.0,
                edge_wave: None,
                sharpen: Some(VHSSharpenSettings::default()),
            });
            cases.push(("vhs", effect));
        }
    }
    for (label, effect) in &cases {
        for (width, height) in [(1, 1), (2, 2), (7, 3), (32, 17), (65, 33)] {
            for (frame_num, scale) in [(0, [1.0, 1.0]), (7, [1.25, 0.75])] {
                compare(&mut gpu, effect, width, height, frame_num, scale, label);
            }
        }
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn stochastic_effects_and_complete_presets_match_cpu() {
    let mut gpu = WgpuBackend::new().expect("a compute adapter is required");
    let mut cases = vec![("complete-default", NtscEffect::default())];
    let mut e = clean_effect();
    e.composite_noise = NtscEffect::default().composite_noise;
    cases.push(("composite-noise", e));
    let mut e = clean_effect();
    e.luma_noise = NtscEffect::default().luma_noise;
    cases.push(("luma-noise", e));
    let mut e = clean_effect();
    e.chroma_noise = NtscEffect::default().chroma_noise;
    cases.push(("chroma-noise", e));
    let mut e = clean_effect();
    e.head_switching = Some(HeadSwitchingSettings::default());
    cases.push(("head-switching", e));
    let mut e = clean_effect();
    e.tracking_noise = Some(TrackingNoiseSettings::default());
    cases.push(("tracking", e));
    let mut e = clean_effect();
    e.snow_intensity = 50.0;
    cases.push(("dense-snow", e));
    let mut e = clean_effect();
    e.chroma_phase_noise_intensity = 0.3;
    cases.push(("phase-noise", e));
    let mut e = clean_effect();
    e.vhs_settings = Some(VHSSettings::default());
    cases.push(("vhs-all", e));
    let mut e = clean_effect();
    e.vhs_settings = Some(VHSSettings {
        tape_speed: VHSTapeSpeed::NONE,
        chroma_loss: 0.2,
        sharpen: None,
        edge_wave: None,
    });
    cases.push(("loss", e));
    for (label, mut effect) in cases {
        for seed in [0, -47, i32::MAX] {
            effect.random_seed = seed;
            for (width, height) in [(7, 3), (65, 33), (257, 67)] {
                for (frame, scale) in [(0, [1.0, 1.0]), (13, [1.25, 0.75])] {
                    compare(&mut gpu, &effect, width, height, frame, scale, label);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn field_modes_and_reused_buffers_match_cpu() {
    use ntsc_rs::gpu::{BackendType, runner::NtscEffectRunner};
    let mut runner = NtscEffectRunner::new(BackendType::Wgpu);
    assert_eq!(runner.active_backend(), BackendType::Wgpu);
    let mut effect = NtscEffect::default();
    effect.scale.as_mut().unwrap().scale_with_video_size = true;
    for field in [
        YiqField::Both,
        YiqField::Upper,
        YiqField::Lower,
        YiqField::InterleavedUpper,
        YiqField::InterleavedLower,
    ] {
        for (width, height) in [(65, 33), (65, 34), (65, 33), (127, 65)] {
            for frame in [0, 1, 17] {
                let mut input = vec![0.0; YiqView::buf_length_for((width, height), field)];
                for (idx, v) in input.iter_mut().enumerate() {
                    *v = ((idx * 17 + frame * 13) % 97) as f32 / 150.0;
                }
                let mut expected = input.clone();
                let mut cpu_view = YiqView::from_parts(&mut expected, (width, height), field);
                effect.apply_effect_to_yiq(&mut cpu_view, frame, [1.25, 0.75]);
                let mut gpu_view = YiqView::from_parts(&mut input, (width, height), field);
                runner.apply_effect(&mut gpu_view, &effect, frame, [1.25, 0.75]);
                assert_eq!(runner.last_backend(), BackendType::Wgpu);
                for (cpu, gpu) in [
                    (cpu_view.y, gpu_view.y),
                    (cpu_view.i, gpu_view.i),
                    (cpu_view.q, gpu_view.q),
                ] {
                    for (&a, &b) in cpu.iter().zip(gpu.iter()) {
                        assert!(
                            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-3,
                            "{field:?} {width}x{height} frame {frame}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn application_wgpu_selection_renders_complete_effect() {
    use ntsc_rs::{
        BackendPreference, apply_effect_to_yiq_with_backend_preference, gpu::BackendType,
    };
    let effect = NtscEffect::default();
    for frame in [0, 3] {
        let mut data = vec![0.2; YiqView::buf_length_for((64, 32), YiqField::InterleavedUpper)];
        let mut view = YiqView::from_parts(&mut data, (64, 32), YiqField::InterleavedUpper);
        let backend = apply_effect_to_yiq_with_backend_preference(
            &effect,
            &mut view,
            frame,
            [1.0, 1.0],
            BackendPreference::Wgpu,
        );
        assert_eq!(backend, BackendType::Wgpu);
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn full_default_rgb_fixture_matches_cpu() {
    use ntsc_rs::{
        gpu::{BackendType, runner::NtscEffectRunner},
        yiq_fielding::{BlitInfo, DeinterlaceMode, Rgb},
    };
    let image = image::load_from_memory(include_bytes!("../benches/balloons.png"))
        .unwrap()
        .to_rgb8();
    let dimensions = (image.width() as usize, image.height() as usize);
    let effect = NtscEffect::default();
    let field = effect.use_field.to_yiq_field(7);
    let blit = BlitInfo::from_full_frame(dimensions.0, dimensions.1, dimensions.0 * 3);
    let mut input = vec![0.0; YiqView::buf_length_for(dimensions, field)];
    YiqView::from_parts(&mut input, dimensions, field).set_from_strided_buffer::<Rgb, u8, _>(
        image.as_raw(),
        blit,
        (),
    );
    let mut outputs = Vec::new();
    for backend in [BackendType::Cpu, BackendType::Wgpu] {
        let mut runner = NtscEffectRunner::new(backend);
        let mut data = input.clone();
        let mut view = YiqView::from_parts(&mut data, dimensions, field);
        runner.apply_effect(&mut view, &effect, 7, [1.0, 1.0]);
        assert_eq!(runner.last_backend(), backend);
        let mut rgb = vec![0; image.as_raw().len()];
        view.write_to_strided_buffer::<Rgb, u8, _>(&mut rgb, blit, DeinterlaceMode::Bob, ());
        outputs.push(rgb);
    }
    let max_error = outputs[0]
        .iter()
        .zip(&outputs[1])
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(
        max_error <= 2,
        "RGB fixture error {max_error}/255 exceeds tolerance"
    );
}
