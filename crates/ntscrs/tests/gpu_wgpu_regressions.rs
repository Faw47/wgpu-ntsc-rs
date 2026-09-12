//! Adapter-backed regressions for repaired WGPU arithmetic and terminal failures.
#![cfg(feature = "gpu-wgpu")]

use ntsc_rs::{
    gpu::{
        GpuBackend,
        wgpu_backend::{WgpuBackend, WgpuBackendError},
    },
    settings::standard::NtscEffect,
    yiq_fielding::{YiqField, YiqView},
};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

fn finite_bits(mut bits: u32) -> u32 {
    if (bits >> 23) & 0xff == 0xff {
        bits ^= 1 << 23;
    }
    bits
}

#[test]
#[ignore = "requires a compute adapter"]
fn software_filter_arithmetic_matches_host_rounding_bit_for_bit() {
    let backend = WgpuBackend::new().expect("adapter required");
    eprintln!("ADAPTER {:?}", backend.adapter_info);

    let filter_source = include_str!("../src/gpu/shaders/filter_plane.wgsl");
    let helper_start = filter_source
        .find("// A four-word unsigned integer")
        .expect("exact-FMA helper marker");
    let helper_end = filter_source
        .find("@compute")
        .expect("filter entry-point marker");
    let shader = format!(
        r#"{}

struct FmaInput {{
    a: u32,
    b: u32,
    c: u32,
    _padding: u32,
}}

@group(0) @binding(0) var<storage, read> inputs: array<FmaInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn fma_test(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= arrayLength(&inputs)) {{ return; }}
    let input = inputs[id.x];
    let a = bitcast<f32>(input.a);
    let b = bitcast<f32>(input.b);
    let c = bitcast<f32>(input.c);
    outputs[id.x] = vec2<u32>(
        bitcast<u32>(upstream_mul_add(a, b, c, true)),
        bitcast<u32>(upstream_mul_add(a, b, c, false)),
    );
}}
"#,
        &filter_source[helper_start..helper_end]
    );

    let mut cases = vec![
        [1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits(), 0],
        [
            f32::MAX.to_bits(),
            2.0f32.to_bits(),
            (-f32::MAX).to_bits(),
            0,
        ],
        [f32::MIN_POSITIVE.to_bits(), 0.5f32.to_bits(), 1u32, 0],
        [1u32, 1u32, 0x8000_0001, 0],
        [0x3f80_0001, 0x3f7f_ffff, (-1.0f32).to_bits(), 0],
    ];
    let mut state = 0x6a09_e667_f3bc_c909u64;
    for _ in 0..16_384 {
        let mut words = [0u32; 4];
        for word in &mut words[..3] {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *word = finite_bits((state ^ (state >> 32)) as u32);
        }
        cases.push(words);
    }

    let expected: Vec<[u32; 2]> = cases
        .iter()
        .map(|case| {
            let a = f32::from_bits(case[0]);
            let b = f32::from_bits(case[1]);
            let c = f32::from_bits(case[2]);
            let rounded_product = std::hint::black_box(a * b);
            [a.mul_add(b, c).to_bits(), (rounded_product + c).to_bits()]
        })
        .collect();
    let input = backend
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("exact FMA test inputs"),
            contents: bytemuck::cast_slice(&cases),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output_size = (expected.len() * size_of::<[u32; 2]>()) as u64;
    let output = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exact FMA test outputs"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("exact FMA test staging"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = backend
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exact FMA property shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
    let pipeline = backend
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("exact FMA property pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("fma_test"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bind_group = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exact FMA property bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
    let mut encoder = backend
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exact FMA property encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("exact FMA property pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
    backend.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    backend
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let actual: &[[u32; 2]] = bytemuck::cast_slice(&mapped);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            actual, expected,
            "case {index}: a={:08x} b={:08x} c={:08x}",
            cases[index][0], cases[index][1], cases[index][2]
        );
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn real_device_loss_remains_visible_across_readbacks() {
    let mut backend = WgpuBackend::new().expect("adapter required");
    eprintln!("ADAPTER {:?}", backend.adapter_info);

    let dimensions = (8, 8);
    let mut source = vec![0.2; YiqView::buf_length_for(dimensions, YiqField::Both)];
    let source = YiqView::from_parts(&mut source, dimensions, YiqField::Both);
    let mut first_frame = backend.upload_frame(&source);
    let mut second_frame = backend.upload_frame(&source);
    backend.apply_effect(&NtscEffect::default(), &mut first_frame, 0, [1.0, 1.0]);
    backend.apply_effect(&NtscEffect::default(), &mut second_frame, 0, [1.0, 1.0]);

    // Queue both readbacks before destroying the device so the test exercises
    // error handling after real submissions, not only the runner's preflight.
    let first_pending = first_frame.enqueue_download();
    let second_pending = second_frame.enqueue_download();
    backend.device.destroy();

    let mut first_output = vec![0.0; YiqView::buf_length_for(dimensions, YiqField::Both)];
    let mut second_output = vec![0.0; YiqView::buf_length_for(dimensions, YiqField::Both)];
    let first_error = first_frame
        .try_finish_download(
            &mut YiqView::from_parts(&mut first_output, dimensions, YiqField::Both),
            first_pending,
        )
        .unwrap_err();
    let second_error = second_frame
        .try_finish_download(
            &mut YiqView::from_parts(&mut second_output, dimensions, YiqField::Both),
            second_pending,
        )
        .unwrap_err();

    assert!(
        matches!(first_error, WgpuBackendError::DeviceLost(_)),
        "first readback did not preserve device loss: {first_error:?}"
    );
    assert!(
        matches!(second_error, WgpuBackendError::DeviceLost(_)),
        "second readback did not preserve device loss: {second_error:?}"
    );
}
