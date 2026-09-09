# GPU parity and performance audit

Baseline: fork commit `80c45e31a306ebe9fb17ea94c4027b32771a7d6e`. Upstream inspected: `88f2df9a27863097eaffd8d1fe0080a174dfd4a5`.

## Problems corrected

The previous benchmark labeled CPU work as GPU work, and the golden test compared CPU output with CPU output. Default interleaved settings triggered CPU fallback. Automatic application selection never selected wgpu; explicit selection recreated the device and pipelines every frame. The workspace also referenced a nonexistent package.

| Area | Applied change | Reason |
| --- | --- | --- |
| Application selection | Enable wgpu in application/plugin defaults; cache a runner per worker thread; return actual backend | Render successive frames on a persistent compute device; expose fallback |
| Fields | Render both interleaved fields with reference order, times, and row counts | Default settings run shaders and preserve odd-height behavior |
| Dispatch | Two-dimensional dispatch for pixel shaders; row-count dispatch for recursive filters | Cover the whole image without launching a row operation per pixel |
| Storage | Separate writable bindings; three-plane scratch; reuse frame and staging allocations | Remove aliasing and out-of-bounds copies |
| Upload and readback | Reuse control buffers and filter bind groups; read back only Y/I/Q; enqueue both fields before waiting | Reduce allocation, transfer, and synchronization overhead |
| Filters | Reference coefficients and initial state; input filters, all demodulators, smear, ringing, composite/tape sharpening, tape lowpass | Preserve filter response and operation order |
| Stochastic effects | Reference row/event random preparation; pixel shaders for noise, shifts, phase, loss, and snow | Preserve random sequence and appearance while moving image processing to compute |
| Snow | Ordered transient lists binned into 32-pixel tiles | Parallel pixel evaluation without scanning all row events or floating-point atomics |
| VHS/head/tracking | Reference row shifts, boundaries, midline transient, phase and all three wave-shifted planes | Correct previously simplified effects |
| Reference edge case | Backport upstream short-row IIR delay-tail fix | Avoid duplicated filter warmup for rows shorter than the delay |
| Tests and benchmarks | Real wgpu execution with required adapter, actual-backend assertions, and numerical comparisons | A CPU fallback cannot pass as a GPU result |

Recursive IIR filters remain sequential within a row. Replacing them with a short FIR blur would change the effect. CPU preparation generates seeds, shifts, masks, and sparse transient control data; it does not render the image planes. This is a hybrid control/compute design, not a claim that every instruction executes on a GPU.

## Validation performed

Rust 1.90.0, Linux, Mesa llvmpipe Vulkan (software adapter):

- 41 test functions passed with `cargo test -p ntsc-rs --features gpu-wgpu -- --include-ignored`, including required-adapter shader tests.
- Deterministic matrix exercises demodulation, filter families, delays, ringing, sharpening and tape speeds at tiny and irregular dimensions.
- Stochastic matrix exercises isolated noise, snow, head switching, tracking, phase, and complete default presets with multiple seeds, frame numbers, scales, and dimensions.
- CPU-only tests also passed. Core-library Clippy completed with no remaining warnings after fixes; two pre-existing warnings remain in the unrelated filter benchmark.
- The existing balloons PNG fixture passes final RGB comparison with maximum channel error at most 2/255 under the full default effect.
- Field tests cover all five field modes, odd/even dimensions, buffer reuse/resizing, scale-with-video-size, and the application's cached selection path. The latter exposed a thread-local destructor ordering crash that was corrected.
- The release benchmark completed progressive and interleaved default presets at 720x480, 1280x720, 1920x1080, and 3840x2160. Every size passed a finite-value and maximum absolute Y/I/Q error gate of 0.002 before measurement.

The short software benchmark measured approximately 41 ms CPU versus 334 ms software-wgpu at progressive 4K, and 39 ms versus 336 ms at interleaved 4K. These are diagnostic software-adapter results, **not hardware GPU speedups**. Automatic selection rejects software adapters. Timing samples were short (10 samples, requested 0.3 s warmup and 0.5 s measurement); use Criterion defaults for hardware measurements.

The reference is this fork's CPU code, with the noted short-row fix. Current upstream has changed RNG and other architecture; these tests do not prove bit identity with current upstream. Absolute float tolerance is a numerical gate, not exhaustive perceptual validation. Native Metal/DX12, real Vulkan GPUs, full applications and host plugins still need validation. This environment lacks the application GStreamer/GTK development dependencies and host SDK setup.

## Remaining work, in priority order

These changes are not the only optimizations needed. Measure the following on the target hardware before claiming faster rendering:

1. **Hardware acceptance gate.** Run the benchmark on representative discrete and integrated GPUs, record adapter/driver/CPU, and require full-preset GPU time below CPU time at the intended resolution. Repeat parity matrices on each driver. Small frames may remain faster on CPU; choose any crossover policy from measurements.
2. **GPU-native preview.** RGB/YIQ conversion still runs on CPU, and preview currently returns pixels to CPU before uploading for display. Share a device with the preview renderer and retain intermediate/output images on GPU to remove those transfers. Compare rendered RGB images as well as YIQ.
3. **Pipelined video export.** The synchronous caller still waits for output each frame. Field readbacks overlap submission, but this is not a multi-frame asynchronous export pipeline. A bounded queue and staging ring must preserve ordering, cancellation and backpressure while overlapping decoding, compute, readback and encoding.
4. **Profile recursive filters and memory traffic.** Use GPU timestamp queries and hardware profiling to identify dominant passes. Evaluate row layout, workgroup shape, safe pass fusion, and ping-pong scratch ownership. Preserve recurrence, edge conditions, and effect order; do not shorten filters to manufacture speedups.
5. **Control preparation and cache pressure.** Reference random generation remains CPU work, including sparse snow amplitudes and row wave samples. Measure it separately before porting exact integer/RNG operations. Per-thread devices and caches can be expensive in hosts with many workers; investigate a bounded per-stream/device pool.
6. **Capacity and recovery.** Validate dense snow, extreme settings, oversized images, adapter storage limits, allocation failure and device loss. A 4K default benchmark is not a guarantee for every legal parameter combination or resolution. Add explicit limits and diagnostics before extending supported sizes.
7. **Upstream and application acceptance.** Decide whether to adopt upstream's newer RNG as a deliberate compatibility change. Complete native application/plugin builds and visual comparisons on footage, gradients, text and saturated colors. The current numerical tests cannot prove every effect combination is perceptually unchanged.

Use the commands in the README. Initialization failure is a test failure in the explicitly requested GPU suite; normal CPU-only tests remain runnable without a compute adapter. The CI shader job uses software Vulkan for correctness only.
