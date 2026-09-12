//! cargo run -p ntsc-rs --features gpu-wgpu --release --example gpu_diagnostics -- 1920 1080
#[cfg(feature = "gpu-wgpu")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ntsc_rs::{
        NtscEffect,
        gpu::{
            GpuBackend,
            block_filter::{BlockFilterExperiment, TransferFunction},
            wgpu_backend::WgpuBackend,
        },
        yiq_fielding::{YiqField, YiqView},
    };
    use std::time::Instant;
    let args: Vec<_> = std::env::args().skip(1).collect();
    let software = args.iter().any(|a| a == "--allow-software");
    let sizes: Vec<_> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let width: usize = sizes
        .first()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1920);
    let height: usize = sizes.get(1).map(|s| s.parse()).transpose()?.unwrap_or(1080);
    let mut gpu = WgpuBackend::try_new()?;
    println!(
        "Adapter: {} | {:?} | {:?}",
        gpu.adapter_info.name, gpu.adapter_info.device_type, gpu.adapter_info.backend
    );
    if gpu.adapter_info.device_type == wgpu::DeviceType::Cpu && !software {
        return Err("Software adapter refused. Use --allow-software only for correctness diagnostics, never hardware performance claims.".into());
    }
    gpu.frame_capacity_requirements(width, height)?;
    let mut data = vec![0.0; YiqView::buf_length_for((width, height), YiqField::Both)];
    {
        let v = YiqView::from_parts(&mut data, (width, height), YiqField::Both);
        for i in 0..width * height {
            v.y[i] = ((i * 13 + i / width * 7) % 256) as f32 / 255.0;
            v.i[i] = ((i * 5 + 11) % 97) as f32 / 200.0 - 0.24;
            v.q[i] = ((i * 3 + 17) % 89) as f32 / 200.0 - 0.22;
        }
    }
    let mut expected = data.clone();
    let effect = NtscEffect::default();
    let start = Instant::now();
    effect.apply_effect_to_yiq(
        &mut YiqView::from_parts(&mut expected, (width, height), YiqField::Both),
        7,
        [1.0, 1.0],
    );
    let cpu_ms = start.elapsed().as_secs_f64() * 1000.0;
    let view = YiqView::from_parts(&mut data, (width, height), YiqField::Both);
    let mut frame = gpu.upload_frame(&view);
    // Warm pipelines before collecting a sample.
    gpu.apply_effect(&effect, &mut frame, 7, [1.0, 1.0]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    let profiling = gpu.set_profiling(true);
    let start = Instant::now();
    gpu.upload_into(&view, &mut frame);
    gpu.begin_execution();
    gpu.apply_effect(&effect, &mut frame, 7, [1.0, 1.0]);
    if let Some(e) = gpu.take_pending_error() {
        return Err(e.into());
    }
    let mut actual = vec![0.0; data.len()];
    frame.try_finish_download(
        &mut YiqView::from_parts(&mut actual, (width, height), YiqField::Both),
        frame.enqueue_download(),
    )?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let mut max = 0.0f32;
    for (a, b) in expected[..width * height * 3]
        .iter()
        .zip(&actual[..width * height * 3])
    {
        assert!(a.is_finite() && b.is_finite());
        max = max.max((a - b).abs());
    }
    assert!(max <= 2e-3, "full effect parity failed: {max}");
    println!(
        "{width}x{height} progressive, max error={max}; CPU={cpu_ms:.3} ms; WGPU host preparation + upload + effect + readback={elapsed:.3} ms"
    );
    if profiling {
        let timings = gpu.read_pass_timings()?.unwrap();
        assert_eq!(timings.len(), gpu.execution_evidence().0.len());
        for (i, t) in timings.iter().enumerate() {
            println!("{:02} {:32} {:10.4} ms", i + 1, t.stage, t.milliseconds);
        }
    } else {
        println!("Adapter does not support timestamp queries.");
    }
    gpu.set_profiling(false);
    let tf = TransferFunction::new(
        &[0.06745527, 0.13491055, 0.06745527],
        &[-1.1429805, 0.4128016],
    );
    let input = &data[..width * height];
    let experiment = BlockFilterExperiment::new(&gpu, &tf, input, width, 1, true)?;
    let mut expected = input.to_vec();
    for row in expected.chunks_exact_mut(width) {
        let initial = row[0];
        tf.filter_signal_in_place(fearless_simd::Level::new(), &mut [row], [initial], 1);
    }
    for parallel in [false, true] {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        experiment.encode(&mut encoder, parallel);
        gpu.queue.submit(Some(encoder.finish()));
        let out = experiment.read(&gpu)?;
        let mut max = 0.0f32;
        for (a, b) in expected.iter().zip(&out) {
            assert!(a.is_finite() && b.is_finite());
            max = max.max((a - b).abs());
        }
        assert!(max <= 2e-3, "block experiment parity failed: {max}");
        let start = Instant::now();
        for _ in 0..10 {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            experiment.encode(&mut encoder, parallel);
            gpu.queue.submit(Some(encoder.finish()));
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        println!(
            "Experimental {}: {:.3} ms/frame (host submit + GPU, no transfers), max={max}",
            if parallel {
                "block biquad"
            } else {
                "native-float serial control"
            },
            start.elapsed().as_secs_f64() * 100.0
        );
    }
    println!(
        "Experimental timings are not production speedups. The production filter also guarantees host-specific rounding."
    );
    Ok(())
}
#[cfg(not(feature = "gpu-wgpu"))]
fn main() {
    eprintln!("Enable --features gpu-wgpu");
    std::process::exit(1);
}
