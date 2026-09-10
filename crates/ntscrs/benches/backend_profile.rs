use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ntsc_rs::{
    NtscEffect,
    gpu::{BackendType, runner::NtscEffectRunner},
    settings::standard::UseField,
    yiq_fielding::{YiqField, YiqView},
};

fn input(width: usize, height: usize) -> Vec<f32> {
    let mut data = vec![0.0; YiqView::buf_length_for((width, height), YiqField::Both)];
    let view = YiqView::from_parts(&mut data, (width, height), YiqField::Both);
    for idx in 0..width * height {
        view.y[idx] = ((idx * 13 + idx / width * 7) % 256) as f32 / 255.0;
        view.i[idx] = ((idx * 5 + 11) % 97) as f32 / 200.0 - 0.24;
        view.q[idx] = ((idx * 3 + 17) % 89) as f32 / 200.0 - 0.22;
    }
    data
}

fn render(
    runner: &mut NtscEffectRunner,
    data: &mut [f32],
    effect: &NtscEffect,
    width: usize,
    height: usize,
) {
    let mut view = YiqView::from_parts(data, (width, height), effect.use_field.to_yiq_field(7));
    runner.apply_effect(&mut view, effect, 7, [1.0, 1.0]);
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut effect = NtscEffect::default();
    let mut cpu = NtscEffectRunner::new(BackendType::Cpu);
    #[cfg(feature = "gpu-wgpu")]
    let mut gpu = {
        let runner = NtscEffectRunner::new(BackendType::Wgpu);
        assert_eq!(
            runner.active_backend(),
            BackendType::Wgpu,
            "GPU benchmark requires a compute adapter; refusing CPU fallback"
        );
        runner
    };
    for (name, field) in [
        ("progressive", UseField::Both),
        ("interleaved", UseField::InterleavedUpper),
    ] {
        effect.use_field = field;
        let mut group = c.benchmark_group(format!("backend/yiq/full-default/{name}"));
        for (width, height) in [(720, 480), (1280, 720), (1920, 1080), (3840, 2160)] {
            let input = input(width, height);
            let resolution = format!("{width}x{height}");
            #[cfg(feature = "gpu-wgpu")]
            {
                // Run real GPU work and check correctness before measuring throughput.
                let mut expected = input.clone();
                let mut actual = input.clone();
                render(&mut cpu, &mut expected, &effect, width, height);
                render(&mut gpu, &mut actual, &effect, width, height);
                assert_eq!(
                    gpu.last_backend(),
                    BackendType::Wgpu,
                    "GPU benchmark fell back: {:?}",
                    gpu.fallback_reason()
                );
                let pixels = width * height;
                for (a, b) in expected[..pixels * 3].iter().zip(&actual[..pixels * 3]) {
                    assert!(
                        a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-3,
                        "benchmark parity gate failed at {resolution}: {a} versus {b}"
                    );
                }

                // Report a reused-buffer sample separately from Criterion's end-to-end result.
                // GPU execution is asynchronous, so compute time remains part of the readback wait
                // unless the adapter supports timestamp-query profiling.
                actual.copy_from_slice(&input);
                render(&mut gpu, &mut actual, &effect, width, height);
                let timings = gpu.last_timings();
                eprintln!(
                    "wgpu host stages {name}/{resolution}: upload={:?} control={:?} encode={:?} submit={:?} readback+gpu-wait={:?} total={:?}",
                    timings.upload,
                    timings.control_preparation,
                    timings.command_encoding,
                    timings.queue_submission,
                    timings.readback_wait_and_copy,
                    timings.total,
                );
            }
            group.bench_with_input(BenchmarkId::new("cpu", &resolution), &input, |b, input| {
                b.iter_batched_ref(
                    || input.clone(),
                    |data| {
                        render(&mut cpu, data, &effect, width, height);
                        black_box(data);
                    },
                    BatchSize::LargeInput,
                );
            });
            #[cfg(feature = "gpu-wgpu")]
            group.bench_with_input(BenchmarkId::new("wgpu", &resolution), &input, |b, input| {
                b.iter_batched_ref(
                    || input.clone(),
                    |data| {
                        render(&mut gpu, data, &effect, width, height);
                        assert_eq!(gpu.last_backend(), BackendType::Wgpu);
                        black_box(data);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
        group.finish();
    }
}

criterion_group!(name = benches; config = Criterion::default(); targets = criterion_benchmark);
criterion_main!(benches);
