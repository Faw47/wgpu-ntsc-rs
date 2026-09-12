#![cfg(feature = "gpu-wgpu")]
use ntsc_rs::gpu::{
    block_filter::{BlockFilterExperiment, TransferFunction, cpu_block_filter},
    wgpu_backend::WgpuBackend,
};

fn filters() -> [TransferFunction; 3] {
    [
        TransferFunction::new(&[0.15, 0.15], &[-0.7]),
        TransferFunction::new(
            &[0.06745527, 0.13491055, 0.06745527],
            &[-1.1429805, 0.4128016],
        ),
        TransferFunction::new(&[0.75, 0.0, 0.75], &[0.0, 0.5]),
    ]
}
fn input(width: usize) -> Vec<f32> {
    let mut state = 0x12345678u32;
    (0..width)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        })
        .collect()
}
fn reference(tf: &TransferFunction, input: &[f32], delay: usize, initial: f32) -> Vec<f32> {
    let mut out = input.to_vec();
    tf.filter_signal_in_place(
        fearless_simd::Level::new(),
        &mut [&mut out],
        [initial],
        delay,
    );
    out
}
fn compare(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            assert!(a.is_finite() && b.is_finite());
            (a - b).abs()
        })
        .fold(0.0f32, f32::max);
    assert!(max <= 2e-3, "max={max}");
}
#[test]
fn cpu_block_formulation_preserves_reference_boundaries_and_delays() {
    for tf in filters() {
        for width in [1, 2, 63, 64, 65, 127, 128, 129, 1919, 1920, 3841, 8192] {
            for data in [input(width), vec![0.4; width], {
                let mut v = vec![0.0; width];
                v[0] = 1.0;
                v
            }] {
                for delay in [0, 1, 3, width, width + 1] {
                    for initial in [0.0, data[0], -0.25] {
                        compare(
                            &reference(&tf, &data, delay, initial),
                            &cpu_block_filter(&tf, &data, delay, initial).unwrap(),
                        );
                    }
                }
            }
        }
    }
    assert!(cpu_block_filter(&filters()[0], &[], 0, 0.0).is_err());
    assert!(cpu_block_filter(&filters()[0], &[1.0], usize::MAX, 0.0).is_err());
    assert!(cpu_block_filter(&TransferFunction::new(&[1.0], &[]), &[1.0], 0, 0.0).is_err());
}
#[test]
#[ignore = "requires a compute adapter"]
fn gpu_block_and_serial_match_upstream_across_block_edges() {
    let backend = WgpuBackend::try_new().expect("compute adapter required");
    for tf in filters() {
        for width in [1, 63, 64, 65, 129, 1920, 8192] {
            for delay in [0, 1, width + 1] {
                for first in [false, true] {
                    let data = input(width);
                    let expected = reference(&tf, &data, delay, if first { data[0] } else { 0.0 });
                    let rows: Vec<_> = (0..3).flat_map(|_| data.iter().copied()).collect();
                    let experiment =
                        BlockFilterExperiment::new(&backend, &tf, &rows, width, delay, first)
                            .unwrap();
                    for parallel in [false, true] {
                        let mut encoder =
                            backend.device.create_command_encoder(&Default::default());
                        experiment.encode(&mut encoder, parallel);
                        backend.queue.submit(Some(encoder.finish()));
                        let out = experiment.read(&backend).unwrap();
                        for row in out.chunks_exact(width) {
                            compare(&expected, row);
                        }
                    }
                }
            }
        }
    }
}
