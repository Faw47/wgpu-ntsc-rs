# GPU profiling runbook

The GPU backend is currently an opt-in implementation and the correctness suite
must pass before timing it. A run on Mesa llvmpipe is useful for exercising the
adapter-backed path, but it is a software run and must not be presented as GPU
performance.

## Correctness gate

From the repository root:

```bash
cargo test -p ntsc-rs --features gpu-wgpu -- --include-ignored
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
