#[cfg(all(test, feature = "gpu-wgpu"))]
mod tests {
    use crate::{
        gpu::{BackendType, runner::NtscEffectRunner},
        settings::standard::NtscEffect,
        yiq_fielding::{Rgbx, YiqField, YiqOwned, YiqView},
    };

    #[cfg(feature = "gpu-wgpu")]
    fn max_plane_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                assert!(x.is_finite() && y.is_finite(), "non-finite plane value");
                (x - y).abs()
            })
            .fold(0.0f32, f32::max)
    }

    #[test]
    #[should_panic(expected = "non-finite plane value")]
    fn max_plane_diff_rejects_nan() {
        max_plane_diff(&[f32::NAN], &[0.0]);
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    #[ignore = "requires a compute adapter"]
    fn default_effect_runs_on_gpu() {
        let width = 4;
        let height = 4;
        let mut pixels: Vec<f32> = vec![0.0; width * height * 4];
        for i in 0..(width * height) {
            pixels[i * 4] = i as f32 / 16.0;
            pixels[i * 4 + 1] = i as f32 / 16.0;
            pixels[i * 4 + 2] = i as f32 / 16.0;
            pixels[i * 4 + 3] = 1.0;
        }

        let mut cpu_yiq = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            &pixels,
            width * 4 * std::mem::size_of::<f32>(),
            width,
            height,
            YiqField::Both,
        );
        let mut cpu_yiq_view = YiqView::from(&mut cpu_yiq);

        let effect = NtscEffect::default();

        let mut cpu_runner = NtscEffectRunner::new(BackendType::Cpu);
        cpu_runner
            .apply_effect(&mut cpu_yiq_view, &effect, 0, [1.0, 1.0])
            .unwrap();

        #[cfg(feature = "gpu-wgpu")]
        {
            let mut yiq = YiqOwned::from_strided_buffer::<Rgbx, f32>(
                &pixels,
                width * 4 * std::mem::size_of::<f32>(),
                width,
                height,
                YiqField::Both,
            );
            let mut yiq_view = YiqView::from(&mut yiq);
            let mut wgpu_runner = NtscEffectRunner::new(BackendType::Wgpu);
            assert_eq!(wgpu_runner.active_backend(), BackendType::Wgpu);
            {
                let execution = wgpu_runner
                    .apply_effect(&mut yiq_view, &effect, 0, [1.0, 1.0])
                    .unwrap();
                assert_eq!(execution.actual, BackendType::Wgpu);
                assert_eq!(execution.cpu_image_effect_invocations, 0);
                assert!(!execution.dispatched_stages.is_empty());
                assert_eq!(wgpu_runner.last_backend(), BackendType::Wgpu);
                assert!(wgpu_runner.fallback_reason().is_none());

                assert_eq!(yiq_view.dimensions, cpu_yiq_view.dimensions);
                const TOL: f32 = 2e-3;
                assert!(
                    max_plane_diff(yiq_view.y, cpu_yiq_view.y) < TOL,
                    "Y plane max diff {}",
                    max_plane_diff(yiq_view.y, cpu_yiq_view.y)
                );
                assert!(
                    max_plane_diff(yiq_view.i, cpu_yiq_view.i) < TOL,
                    "I plane max diff {}",
                    max_plane_diff(yiq_view.i, cpu_yiq_view.i)
                );
                assert!(
                    max_plane_diff(yiq_view.q, cpu_yiq_view.q) < TOL,
                    "Q plane max diff {}",
                    max_plane_diff(yiq_view.q, cpu_yiq_view.q)
                );
            }
        }
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    #[ignore = "requires a compute adapter"]
    fn interleaved_field_wgpu_runner_matches_cpu_reference() {
        let width = 8;
        let height = 8;
        let mut pixels: Vec<f32> = vec![0.0; width * height * 4];
        for i in 0..(width * height) {
            let v = (i % 17) as f32 / 17.0;
            pixels[i * 4] = v;
            pixels[i * 4 + 1] = v * 0.8;
            pixels[i * 4 + 2] = v * 0.6;
            pixels[i * 4 + 3] = 1.0;
        }

        let effect = NtscEffect::default();

        let mut direct = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            &pixels,
            width * 4 * std::mem::size_of::<f32>(),
            width,
            height,
            YiqField::InterleavedUpper,
        );
        let mut direct_view = YiqView::from(&mut direct);
        effect.apply_effect_to_yiq(&mut direct_view, 0, [1.0, 1.0]);

        let mut runner_yiq = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            &pixels,
            width * 4 * std::mem::size_of::<f32>(),
            width,
            height,
            YiqField::InterleavedUpper,
        );
        let mut runner_view = YiqView::from(&mut runner_yiq);
        let mut wgpu_runner = NtscEffectRunner::new(BackendType::Wgpu);
        assert_eq!(wgpu_runner.active_backend(), BackendType::Wgpu);
        {
            let execution = wgpu_runner
                .apply_effect(&mut runner_view, &effect, 0, [1.0, 1.0])
                .unwrap();
            assert_eq!(execution.actual, BackendType::Wgpu);
            assert_eq!(execution.cpu_image_effect_invocations, 0);
            assert!(!execution.dispatched_stages.is_empty());
            assert_eq!(wgpu_runner.last_backend(), BackendType::Wgpu);
            assert!(wgpu_runner.fallback_reason().is_none());
            assert_eq!(direct_view.y.len(), runner_view.y.len());
            assert!(
                max_plane_diff(direct_view.y, runner_view.y) < 2e-3,
                "interleaved Y mismatch"
            );
            assert!(
                max_plane_diff(direct_view.i, runner_view.i) < 2e-3,
                "interleaved I mismatch"
            );
            assert!(
                max_plane_diff(direct_view.q, runner_view.q) < 2e-3,
                "interleaved Q mismatch"
            );
        }
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    #[ignore = "requires a compute adapter"]
    fn direct_rgba8_output_matches_cpu_write_path() {
        let width = 8;
        let height = 8;
        let mut pixels = vec![0.0f32; width * height * 4];
        for (index, pixel) in pixels.chunks_exact_mut(4).enumerate() {
            let value = (index % 19) as f32 / 19.0;
            pixel[0] = value;
            pixel[1] = value * 0.7;
            pixel[2] = value * 0.4;
            pixel[3] = 1.0;
        }
        let effect = NtscEffect::default();
        let mut cpu = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            &pixels,
            width * 4 * std::mem::size_of::<f32>(),
            width,
            height,
            YiqField::Both,
        );
        let mut cpu_view = YiqView::from(&mut cpu);
        effect.apply_effect_to_yiq(&mut cpu_view, 0, [1.0, 1.0]);
        let mut expected = vec![0u8; width * height * 4];
        cpu_view.write_to_strided_buffer::<Rgbx, u8, _>(
            &mut expected,
            crate::yiq_fielding::BlitInfo::from_full_frame(width, height, width * 4),
            crate::yiq_fielding::DeinterlaceMode::Bob,
            (),
        );

        let mut source = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            &pixels,
            width * 4 * std::mem::size_of::<f32>(),
            width,
            height,
            YiqField::Both,
        );
        let source_view = YiqView::from(&mut source);
        let mut output = vec![0u8; expected.len()];
        let mut runner = NtscEffectRunner::new(BackendType::Wgpu);
        let execution = runner
            .apply_effect_to_rgba8(&source_view, &effect, 0, [1.0, 1.0], &mut output)
            .unwrap();
        assert_eq!(execution.actual, BackendType::Wgpu);
        assert_eq!(output, expected);
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    #[ignore = "requires a compute adapter"]
    fn wgpu_runners_share_immutable_context() {
        let first =
            crate::gpu::wgpu_backend::WgpuBackend::try_new().expect("compute adapter required");
        let second =
            crate::gpu::wgpu_backend::WgpuBackend::try_new().expect("compute adapter required");
        assert!(first.shares_context_with(&second));
        assert!(std::sync::Arc::ptr_eq(&first.device, &second.device));
        assert!(std::sync::Arc::ptr_eq(&first.queue, &second.queue));
    }

    #[test]
    #[ignore = "requires a compute adapter"]
    fn destroyed_explicit_wgpu_device_returns_an_error_without_cpu_processing() {
        let effect = NtscEffect::default();
        let mut runner = NtscEffectRunner::new(BackendType::Wgpu);
        assert_eq!(runner.active_backend(), BackendType::Wgpu);
        runner.destroy_wgpu_device_for_test();

        let dimensions = (8, 8);
        let mut data = vec![0.2; YiqView::buf_length_for(dimensions, YiqField::Both)];
        let original = data.clone();
        for _ in 0..2 {
            let error = runner
                .apply_effect(
                    &mut YiqView::from_parts(&mut data, dimensions, YiqField::Both),
                    &effect,
                    0,
                    [1.0, 1.0],
                )
                .unwrap_err();
            assert_eq!(error.kind, crate::gpu::BackendFailureKind::DeviceLost);
        }
        assert_eq!(
            data, original,
            "repeated explicit WGPU failures must not invoke the CPU effect"
        );
    }
}
