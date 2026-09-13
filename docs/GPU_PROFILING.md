# GPU profiling runbook

The GPU backend is currently an opt-in implementation and the correctness suite
must pass before timing it. A run on Mesa llvmpipe is useful for exercising the
adapter-backed path, but it is a software run and must not be presented as GPU
performance.

## Correctness gate

From the repository root:

```bash
cargo test -p ntsc-rs --features gpu-wgpu -- --include-ignored \
  --skip pinned_upstream_default_fixed_seed_fingerprint \
  --skip fixed_seed_stochastic_control_fingerprints
```

The suite checks CPU/GPU parity, WGSL validation, backend selection, and the
wide recursive-filter regressions. A backend that cannot create an explicit
adapter fails rather than silently falling back to the CPU renderer.

## Pass diagnostics

The diagnostics example prints the adapter identity and separates CPU
preparation, upload, compute, readback, and total wall-clock time. Pass-level
timestamp queries are enabled with `NTSC_GPU_PROFILE=1`:

```bash
NTSC_GPU_PROFILE=1 cargo run -p ntsc-rs --features gpu-wgpu \
  --example gpu_diagnostics -- 1920 1080
```

Use `--allow-software` only when you intentionally want to exercise a software
Vulkan adapter in CI or a headless environment:

```bash
cargo run -p ntsc-rs --features gpu-wgpu \
  --example gpu_diagnostics -- 128 72 --allow-software
```

The output labels software adapters explicitly. Do not use those timings to
decide whether the RX 6800 path is worthwhile.

The strict shader path uses integer-emulated binary32 add and multiply-add
rounding so that the GPU remains within the CPU parity target. For throughput
experiments, set `NTSC_WGPU_FAST_MATH=1` before starting the process. This
selects native adapter f32 arithmetic for recursive filters and demodulation.
Fast math is intentionally not a bit-exact parity mode, and the default is
unchanged when the variable is absent.

The recursive row filter defaults to a 64-thread workgroup. Hardware tuning
can select `NTSC_WGPU_FILTER_WORKGROUP=32`, `64`, `128`, or `256`; unsupported
values or sizes beyond the adapter limit fall back to 64. This changes only
dispatch shape, not filter arithmetic, and must be benchmarked on the target
adapter.

For the desktop preview path, `NTSC_GPU_PROFILE_HOST=1` logs the CPU input
conversion, backend call, output conversion, and total time for each processed
frame. This is useful for separating RGB/YIQ conversion and UI-side copies
from WGPU work. The setting is sampled once per process and should be disabled
for normal playback.

The preview can also opt into adapter-side YIQ-to-RGBA8 conversion with
`NTSC_WGPU_DIRECT_RGBA8=1`. This path is intentionally narrow: it applies only
to progressive, tightly packed 8-bit previews using a full-frame `Both` field.
Fielded, cropped, split-screen, and high-bit-depth output continue through the
general CPU write path so their semantics do not change. If automatic backend
selection cannot create WGPU, the preview retries through its normal CPU path.

Library integrations that own their frame queue can use
`WgpuBackend::apply_effect_async` to submit an effect and enqueue its readback
without waiting. Keep a bounded number of frame slots, then finish pending
readbacks in order. The desktop GStreamer transform remains synchronous, so it
uses this capability only for batching the active interlaced fields today.

For interactive playback, `NTSC_PREVIEW_LOW_LATENCY=1` makes the decoded video
queue bounded and leaky downstream. That setting is deliberately opt-in and is
not applied to render jobs, where dropping frames would be incorrect.

## Hardware benchmark

On the target machine, first run the correctness gate and then the end-to-end
benchmark:

```bash
cargo bench -p ntsc-rs --features gpu-wgpu --bench backend_profile
```

The benchmark refuses a software adapter unless `NTSC_ALLOW_SOFTWARE_GPU=1`
is set. It covers progressive and interlaced frames at 480p, 720p, 1080p, and
4K, including preparation, upload, compute, and readback. Record the adapter,
driver, resolution, and whether the run used the production row filter before
making an architectural decision.

The parity gate also prints WGPU host stages for each progressive and
interlaced sample: independent control preparation, command encoding, and queue
submission. The control streams use independent reference seeds and are
prepared concurrently; the stage-level parity test keeps their generated data
identical to the serial helpers. Use a physical adapter run to decide whether
the extra host parallelism offsets its allocation cost on the target workload.

## Experimental block filter

`gpu::block_filter` is isolated from normal rendering. It demonstrates
affine-state block propagation for a representative low-order IIR and includes
CPU and WGPU parity tests. It is not enabled in the production pipeline:

```bash
cargo test -p ntsc-rs --features gpu-wgpu --test block_filter \
  -- --include-ignored
```

The formulation changes floating-point operation ordering. It must therefore
be compared against the reference at every supported width, delay, filter
coefficient set, and target adapter before it can replace the current path.
The design follows the boundary-state/prefix approach described by
[gpufilter](https://github.com/andmax/gpufilter), without claiming parity or
performance from that project.

Until a physical RX 6800 (or another target adapter) supplies timings, keep
the current production row filter and report software-adapter results as
correctness evidence only.
