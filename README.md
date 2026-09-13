<p align="center">
    <a href="https://ntsc.rs">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="./docs/img/logo-darkmode.svg">
            <img alt="ntsc-rs logo" src="./docs/img/logo-lightmode.svg" width="1216">
        </picture>
    </a>
</p>

---

**ntsc-rs** is a video effect which emulates NTSC and VHS video artifacts. It can be used as an After Effects, Premiere, or OpenFX plugin, or as a standalone application.

![Screenshot of the ntsc-rs standalone application](./docs/img/appdemo.png)

## About this fork

This repository is derived from **[ntsc-rs](https://github.com/ntsc-rs/ntsc-rs)**. The original project, its algorithms, and naming remain the work of that upstream team; this fork focuses on GPU acceleration and UI modernization.

### GPU implementation and validation

The `gpu-wgpu` feature enables Vulkan, Metal, or DX12 compute rendering and is enabled by default in the standalone application and plugin manifests. Automatic selection uses a hardware adapter when available and otherwise uses the CPU reference. Explicit `wgpu` selection also permits software adapters for validation.

The effect pipeline now includes the reference filters, demodulation modes, noise, snow, head switching, tracking, VHS processing, and interleaved fields. GPU resources are reused across frames. Rust prepares reference random control data; compute shaders process the image planes.

**Validation is bounded, not a universal parity or speed guarantee.** The CPU reference now follows pinned current-upstream SplitMix64/Mix4 randomness and proxy-scaling semantics, with fixed-seed regression coverage. The adapter-backed CPU/GPU suite retains an absolute Y/I/Q error threshold of 0.002, but hardware GPU performance and cross-driver parity still require measurement; the former 10x speedup claim was not supported by the old benchmark. See `GPU_PORT_SPEC.md` for the exact verified and hardware-gated scope.

Run the real shader tests and end-to-end YIQ benchmark on the target machine:

```sh
cargo test -p ntsc-rs --features gpu-wgpu -- --include-ignored \
  --skip pinned_upstream_default_fixed_seed_fingerprint \
  --skip fixed_seed_stochastic_control_fingerprints
cargo bench -p ntsc-rs --features gpu-wgpu --bench backend_profile
```

The benchmark includes control preparation, upload, compute, and readback, checks output parity, and refuses CPU fallback. It excludes device initialization, RGB conversion, and video decoding/encoding. Check the printed adapter name: software Vulkan timings do not establish hardware acceleration.

See [the GPU audit](docs/gpu-audit.md) for implemented changes, validation limits, and remaining optimization work.

Performance tuning is opt-in so the default remains the strict parity path:

```sh
# Native adapter f32 arithmetic for filter and demodulation experiments.
NTSC_WGPU_FAST_MATH=1 cargo run -p ntsc-rs-gui --release

# Try a different recursive-filter row workgroup on the target adapter.
NTSC_WGPU_FILTER_WORKGROUP=128 cargo run -p ntsc-rs-gui --release

# Keep interactive preview latency bounded by dropping stale decoded frames.
NTSC_PREVIEW_LOW_LATENCY=1 cargo run -p ntsc-rs-gui --release

# Convert full-frame progressive Both previews to RGBA8 on the adapter.
NTSC_WGPU_DIRECT_RGBA8=1 cargo run -p ntsc-rs-gui --release
```

`NTSC_WGPU_FAST_MATH` is not a bit-exact mode and must be measured against the
strict default. `NTSC_GPU_PROFILE_HOST=1` adds per-frame input conversion,
backend, output conversion, and total timing to the desktop preview log.
`NTSC_WGPU_DIRECT_RGBA8` is a narrow preview optimization for tightly packed
progressive 8-bit `Both` frames; fielded, cropped, and high-bit-depth output
keep the general path.
8-bit exports now keep an 8-bit filter output; 10/12-bit H.264 and FFV1
exports retain Argb64 processing.

## A Note on Development

This fork has been developed with substantial language-model assistance. Its behavior and performance should be assessed from reproducible tests and measurements.

## Download and Install

The latest version of ntsc-rs can be downloaded from [the releases page](https://github.com/valadaptive/ntsc-rs/releases).

After downloading, [read the documentation for how to run it](https://ntsc.rs/docs/standalone-installation/). In particular, ntsc-rs will not work properly on Linux unless you install all of the GStreamer packages listed in the documentation.

## More information

ntsc-rs is a rough Rust port of [ntscqt](https://github.com/JargeZ/ntscqt), a PyQt-based GUI for [ntsc](https://github.com/zhuker/ntsc), itself a Python port of [composite-video-simulator](https://github.com/joncampbell123/composite-video-simulator). Reimplementing the image processing in multithreaded Rust allows it to run at (mostly) real-time speeds.

It's not an exact port--some processing passes have visibly different results, and some new ones have been added.
