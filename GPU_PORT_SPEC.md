# GPU Port Specification

## Status and Scope

This is a living, source-grounded specification. It was repaired and partially implemented on branch `codex/upstream-parity-repair` from fork baseline `7aa7bd7e170f122de78a60507ed87de98f8d2ba3`, targeting upstream `88f2df9a27863097eaffd8d1fe0080a174dfd4a5`.

The implementation pass directly read upstream's complete `NtscEffect::apply_effect_to_yiq_field`, inspected every core commit since the merge base, read the complete WGPU test file and every live WGSL module, and traced GUI preview, GUI render/export, CLI, After Effects, and OpenFX routing. The previous audit's contrary scope disclaimers are superseded by the concrete reconciliation below.

Implemented in this pass: current-upstream SplitMix64/Mix4 RNG and stochastic seeding, current-upstream proxy scaling, current-upstream `fearless_simd` 0.7 simplex behavior, deterministic fixed-seed fixtures, row-parallel `prepare::noise` and `prepare::phase`, removal of the dead shader-side Xoshiro branch, adapter-independent WGSL validation, and explicit requested/selected/fallback/submission instrumentation.

Rust 1.90 validation is available in this environment. No Vulkan, Metal, or DX12 adapter is available, so actual WGPU dispatch parity and hardware performance remain explicitly blocked. No result from this environment establishes hardware acceleration.

## Pinned Reference Revisions

| Repository | Role | Pinned SHA | Verification |
|---|---|---|---|
| `ntsc-rs/ntsc-rs` (GitHub org, formerly `valadaptive/ntsc-rs`) | Upstream, authoritative for semantics | `88f2df9a27863097eaffd8d1fe0080a174dfd4a5` | `VERIFIED_FROM_SOURCE`, `git clone https://github.com/ntsc-rs/ntsc-rs.git` this session, `git rev-parse HEAD` |
| `Faw47/wgpu-ntsc-rs` | Fork under audit | `7aa7bd7e170f122de78a60507ed87de98f8d2ba3` | `VERIFIED_FROM_SOURCE`, `git clone` + `git fetch origin` this session, `git rev-parse origin/main`. This is the merge commit for PR #20 ("fix/gpu-parity-validation"), dated the same day as this audit. Re-fetching produced the same SHA as the prior conversational check, so `main` has not advanced since. |
| Common ancestor of the two above | Fork point | `add90f5bf1bf7e3c573e4e945a16f44a82941b51` ("Version 0.9.4", 2026-03-18) | `VERIFIED_FROM_SOURCE`, computed via `git merge-base` after adding the upstream clone as a remote of the fork clone (`git remote add upstream ... && git merge-base HEAD upstream/main`) |

Divergence size: upstream has 49 commits since the fork point; the fork has 46. Of upstream's 49, 22 touch `crates/ntscrs/src` or its `Cargo.toml` (the effect-processing core); the rest are GUI/build/dependency churn. Those 22 are enumerated and classified in `## Upstream Processing Model`.

The target for effect parity is upstream at `88f2df9`. No RNG compatibility exception remains.

## Definition of Done

A stage is done, for the purposes of this specification, only when all of the following hold simultaneously:

1. Its mathematical work executes through a WGPU compute pass when the WGPU backend is selected, not merely through a WGPU-allocated buffer holding a CPU-computed result.
2. Its numerical output matches upstream `88f2df9` within a stated, justified tolerance, or a named, source-evidenced intentional deviation is documented and accepted as a substitute target.
3. Any CPU-side work it still requires (control-data generation, format conversion, orchestration) is named explicitly, not hidden inside "GPU-accelerated."
4. Its behavior is identical, or explicitly and separately verified, across every application entry point that can invoke it (standalone GUI export, GUI preview, After Effects plugin, OpenFX plugin).
5. A test exists that would fail if the stage silently fell back to the CPU implementation while WGPU was the requested backend.

No live WGSL stage is labeled equivalent merely because a shader or dispatch exists. Static equation review and focused tests are recorded separately, and every live shader remains runtime-gated until the ignored WGPU parity suite runs on an adapter.

## Source-of-Truth Policy

Evidence order:

1. Pinned upstream source at `88f2df9`.
2. Pinned fork baseline at `7aa7bd7`.
3. Direct commit and file diffs from merge base `add90f5`.
4. Executed dual-repository fixtures and Rust tests.
5. WGPU execution tests only when an adapter actually exists.

Earlier audit statements that were disproved are not retained as alternatives. In particular, stage order is not derived from the GPU fork, RNG is not an open product choice, deterministic phase error is not stochastic, the SIMD bump is effect-visible, and a proposed frame ring or resident preview is not presumed implementable.

## Upstream Processing Model

### 1. Effect settings surface

`crates/ntscrs/src/settings/standard.rs` defines `NtscEffect` (upstream `88f2df9`, fork `7aa7bd7`: identical field set, `VERIFIED_FROM_SOURCE` via direct field-list diff). It has 18 direct scalar/enum fields and 8 optional sub-effect blocks, for 26 top-level members:

**Direct fields:** `random_seed: i32`, `use_field: UseField`, `filter_type: FilterType`, `input_luma_filter: LumaLowpass`, `chroma_lowpass_in: ChromaLowpass`, `chroma_demodulation: ChromaDemodulationFilter`, `luma_smear: f32`, `composite_sharpening: f32`, `video_scanline_phase_shift: PhaseShift`, `video_scanline_phase_shift_offset: i32`, `snow_intensity: f32`, `snow_anisotropy: f32`, `chroma_phase_noise_intensity: f32`, `chroma_phase_error: f32`, `chroma_delay_horizontal: f32`, `chroma_delay_vertical: i32`, `chroma_vert_blend: bool`, `chroma_lowpass_out: ChromaLowpass`.

**Optional blocks** (upstream: `SettingsBlock<T>`; fork: `Option<T>`, functionally equivalent wrapper, `VERIFIED_FROM_SOURCE`): `head_switching: HeadSwitchingSettings { height, offset, horiz_shift, mid_line: Option<HeadSwitchingMidLineSettings{position, jitter}> }`, `tracking_noise: TrackingNoiseSettings { height, wave_intensity, snow_intensity, snow_anisotropy, noise_intensity }`, `composite_noise: FbmNoiseSettings { frequency, intensity, detail }`, `ringing: RingingSettings { frequency, power, intensity }`, `luma_noise: FbmNoiseSettings`, `chroma_noise: FbmNoiseSettings`, `vhs_settings: VHSSettings { tape_speed: VHSTapeSpeed, chroma_loss, sharpen: Option<VHSSharpenSettings{intensity, frequency}>, edge_wave: Option<VHSEdgeWaveSettings{intensity, speed, frequency, detail}> }`, `scale: ScaleSettings { horizontal_scale, vertical_scale, scale_with_video_size }`.

**Enums:** `UseField { Alternating, Upper, Lower, Both, InterleavedUpper, InterleavedLower }` (6), `FilterType { ConstantK, Butterworth }` (2), `LumaLowpass { None, Box, Notch }` (3), `PhaseShift { Degrees0, Degrees90, Degrees180, Degrees270 }` (4), `VHSTapeSpeed { NONE, SP, LP, EP }` (4), `ChromaLowpass { None, Light, Full }` (3), `ChromaDemodulationFilter { Box, Notch, OneLineComb, TwoLineComb }` (4).

`VERIFIED_FROM_SOURCE` finding: **no settings were added, removed, or renamed upstream since the fork point.** This bounds the parity problem to "does each existing setting compute the same value," not "does an entire feature exist yet."

### 2. Processing order (CPU reference)

The canonical order below was derived directly from upstream `88f2df9` by reading the complete `NtscEffect::apply_effect_to_yiq_field` body and the helpers it invokes. It was then compared, branch by branch and dependency by dependency, with `gpu::wgpu_backend::WgpuBackend::apply_effect`. The order is the same in both. The earlier derivation from the GPU fork was evidentially backwards even though its resulting 23-item order happened to be correct.

1. Resolve caller/proxy scale and optional user scale.
2. Apply the input luma filter.
3. Apply chroma low-pass-in.
4. Modulate chroma into luma unconditionally.
5. Apply composite sharpening.
6. Apply composite FBM noise.
7. Apply top-level snow.
8. Apply head switching.
9. Apply tracking noise.
10. Demodulate luma into chroma.
11. Apply luma smear.
12. Apply ringing.
13. Apply luma FBM noise.
14. Apply chroma FBM noise to I, then Q, using distinct tags.
15. Apply deterministic chroma phase error.
16. Apply stochastic chroma phase noise.
17. Apply horizontal and vertical chroma delay.
18. Apply VHS edge wave.
19. Apply VHS tape-speed luma and chroma filters.
20. Apply VHS chroma loss.
21. Apply VHS sharpen.
22. Apply vertical chroma blend.
23. Apply chroma low-pass-out.

`filter_type` and scanline phase/offset are cross-cutting parameters, not extra stages. The one-row demodulation override, filter initial conditions and delay, interpolation boundaries, field-specific frame numbers, and VHS substage ordering were checked as behavioral dependencies rather than by function-name matching.

### 3. Upstream changes since the fork point, classified

All 22 commits touching `crates/ntscrs/src` or its `Cargo.toml` after merge base `add90f5` were inspected. Potentially semantic hunks were resolved as follows:

| Commit(s) | Resolution |
|---|---|
| `d62ef09` | Semantic proxy-scale fix. It was absent from both fork CPU and WGPU code and is now shared through `effective_scale_factors`, with a focused regression. |
| `2e62530` | Semantic RNG replacement. Current upstream SplitMix64 with the standard 64-bit finalizer and Stafford Mix4 for 32-bit/float values is now the target and implementation. |
| `4284bf1`, `a655965`, `57369f2` | The `fearless_simd` migration was effect-visible for simplex gradients. Merely calling it dependency churn was wrong. The fork now uses 0.7 and upstream grid/gradient semantics. Direct dual-repository runs became bit-for-bit equal only after this migration. |
| `73c695d` | The sqrt-table portion is numerically neutral, but it sits beside the effect-visible SIMD grid migration. The table/constants and current grid form were ported together. |
| `ff8ae76` | The corrected `delay.max(width)..width+delay` loop was already in the fork. A focused `delay >= width` regression was added. |
| `f79cc75` | Full symbol review found a mechanical move into `EffectCtx`; no independent stage, coefficient, ordering, or boundary change. The later scale and RNG commits are treated separately above. |
| `9caf665`, `fbd2702`, `ab8b6fb`, `3f444ec`, `29a41de`, `c6c5717`, `613985f`, `6654769`, `b3af8c6`, `2ebe9fb` | Settings-wrapper, no-std, serialization, naming, benchmark, thread-stack, test, or formatting changes. Direct call-site and conversion review found no visible effect change. |
| `efa3a3f`, `7f9e53b`, `cc1c1aa`, `7a5fd8b` | Dependency-only changes outside the effect math after separating the SIMD changes above. |

A direct fork-versus-upstream executable comparison at 65x33, seed -47, frame 13, and scale [1.25, 0.75] was bit-for-bit equal for: all effects disabled, composite noise only, snow only, head switching only, tracking only, luma noise only, chroma noise only, phase noise only, VHS only, and the full default stack. The full default fingerprint is `b95f09d0ce6af57a` and is now a unit-test fixture generated from the pinned upstream behavior.

## Exhaustive CPU-to-WGPU Parity Matrix

Statuses deliberately distinguish implementation and static review from runtime equivalence:

- `STATIC_MATCH_RUNTIME_GATED`: Rust and WGSL equations, constants, indices, boundaries, workgroup guards, state, and dispatch were compared, and a focused parity case exists, but no adapter was available in this run.
- `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED`: current-upstream control values are produced on CPU and uploaded; image/sample math is WGSL. CPU preparation has deterministic fixtures. Actual dispatch parity remains adapter-gated.
- `HOST_PARAMETER`: no image math belongs on GPU.

| # | Stage | Current classification | Source and test evidence |
|---:|---|---|---|
| 1 | Scale resolution | `HOST_PARAMETER`, fixed | `effective_scale_factors` is shared by CPU/WGPU; disabled-scale proxy regression added. |
| 2 | Input luma filter | `STATIC_MATCH_RUNTIME_GATED` | Box window, IIR coefficients/initial state, row indexing, delay tail, 64x1 guard inspected; deterministic matrix covers None/Box/Notch and small widths. |
| 3 | Chroma low-pass-in | `STATIC_MATCH_RUNTIME_GATED` | CPU-built ConstantK/Butterworth coefficients feed the same recursive WGSL filter; both Light/Full and filter types covered. |
| 4 | Chroma into luma | `STATIC_MATCH_RUNTIME_GATED` | Carrier phase and I/Q multiplier equations inspected against upstream; 16x16 bounds covered. |
| 5 | Composite sharpen | `STATIC_MATCH_RUNTIME_GATED` | Upstream filter construction is CPU-side; recursive application is `filter_plane.wgsl`; isolated case exists. |
| 6 | Composite noise | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | SplitMix64 row controls fixed and parallel/reference tested; simplex hash/gradient/FBM WGSL inspected and its stale 1D sign corrected. |
| 7 | Top-level snow | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | CPU event order, geometric walk, event substreams and tile binning inspected; WGSL evaluates overlapping events in reference order; dense case exists. |
| 8 | Head switching | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | Row and mid-line seed streams, shift interpolation, zero boundary, partial-row copy, transient envelope inspected; isolated case exists. |
| 9 | Tracking noise | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | Initial RNG draws, row-derived noise/snow, intensity ramp, shared event merge, shifts and three dispatches inspected; isolated case exists. |
| 10 | Luma-to-chroma demodulation | `STATIC_MATCH_RUNTIME_GATED` | Box/Notch/one-line/two-line equations, first/last-line reflection, horizontal neighbors, carrier phase and one-row override inspected; all modes and tiny/odd dimensions covered. Edge-neighbor storage reads now use control-flow guards because WGSL `select` evaluates both value operands and did not make the previous out-of-range load expressions safe. |
| 11 | Luma smear | `STATIC_MATCH_RUNTIME_GATED` | Low-pass construction and recursive WGSL application inspected; isolated case exists. |
| 12 | Ringing | `STATIC_MATCH_RUNTIME_GATED` | Band-pass construction, scaling, initial state and recursive application inspected; isolated case exists. |
| 13 | Luma noise | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | Same verified row control and FBM path as stage 6 with tag 10. |
| 14 | Chroma noise I then Q | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | Distinct tags 1 and 9, dispatch order and per-plane bindings inspected; isolated case exists. |
| 15 | Deterministic phase error | `STATIC_MATCH_RUNTIME_GATED` | This stage has no RNG. WGSL rotates I/Q by `intensity * 2π`; stale Xoshiro/Murmur branch removed. |
| 16 | Stochastic phase noise | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | SplitMix64 per-row sine/cosine controls are fixed and parallel/reference tested; WGSL rotation inspected. |
| 17 | Chroma delay | `STATIC_MATCH_RUNTIME_GATED` | Scale/round rules, scratch copies, vertical zero boundary, floor/linear interpolation and horizontal zero boundary inspected; positive/negative cases exist. |
| 18 | VHS edge wave | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | Base seed excludes frame, frame enters 2D temporal coordinate, upstream simplex sampling stays CPU, WGSL shifts all planes with extend boundary. |
| 19 | VHS tape filters | `STATIC_MATCH_RUNTIME_GATED` | Tape-speed coefficients, chroma delay rounding, Y/I/Q order and luma subtraction pass inspected; all speeds and filter types covered. |
| 20 | VHS chroma loss | `CPU_CONTROL_GPU_PIXEL_RUNTIME_GATED` | Sequential SplitMix64 geometric walk fixed; WGSL zeroes marked I/Q rows; isolated case exists. |
| 21 | VHS sharpen | `STATIC_MATCH_RUNTIME_GATED` | Frequency multiplier, scale and Y-only recursive pass inspected; covered with VHS matrix. |
| 22 | Chroma vertical blend | `STATIC_MATCH_RUNTIME_GATED` | One invocation per column, top zero initial state, sequential previous-sample update and 64x1 guard match upstream. |
| 23 | Chroma low-pass-out | `STATIC_MATCH_RUNTIME_GATED` | Same evidence as stage 3, at the verified final position. |
| Cross-cutting | Field routing | `STATIC_MATCH_RUNTIME_GATED` | Both, Upper, Lower, InterleavedUpper/Lower existing coverage was read in full; explicit Alternating frame-parity coverage added. |
| Cross-cutting | RNG core | Fixed, CPU control only | Current-upstream SplitMix64/Mix4 implemented. Fixed u64/u32/f32 sequences and every stochastic control path are locked by tests. |

All eight live shader modules are parsed and validated by Naga without an adapter. That proves WGSL validity, not numerical execution. The ignored WGPU suite remains the required runtime proof.

## Current WGPU Architecture Audit

The live path is `lib.rs::apply_effect_to_yiq_with_backend_preference` -> thread-local `NtscEffectRunner` -> `WgpuBackend`. The fork-only `backend/` scaffolding is not the application path.

Device, pipelines, parameter buffers, filter bind groups, control-data buffers, frame buffers and staging buffers are already persistent/reused. Each active field records the complete effect chain into one compute command buffer and submits once. Readback is a second submission. Interleaved fields use two independent `WgpuFrame` slots, enqueue both readbacks before the first blocking poll, and preserve the upstream doubled field timebase.

This is field-level overlap, not a two-frame export ring. The caller's YIQ-to-RGB conversion and GStreamer/encoder handoff begin only after `finish_download` returns. No cross-frame in-flight ownership contract exists in the shared entry point.

## CPU Involvement in the WGPU Path

| CPU work | Current state | Disposition |
|---|---|---|
| SplitMix64 row controls for composite/luma/chroma/phase noise | Row-independent and now uses the configured crate thread pool | `DO_NOW` complete for `prepare::noise` and `prepare::phase`. |
| Head-switching controls | Independent rows plus one independently seeded mid-line case | Correct and small; parallelization remains `MEASURE_FIRST`. |
| Snow and tracking event buffers | Per-row RNG is independent, but event/tile offsets are merged in row order | Keep current ordered construction unless a measured local-build plus deterministic-merge redesign justifies complexity. |
| VHS chroma loss | One sequential geometric RNG walk | Must remain sequential. |
| VHS edge-wave noise controls | Batched upstream simplex sampling; frame is a coordinate, not a seed | Leave on CPU; cheap tail work is not a priority. |
| Filter coefficients | Recomputed, with GPU bind groups cached by coefficient bits | `MEASURE_FIRST`; quantify CPU time and cache hit rate first. |
| YIQ upload and readback | Three queue writes, one staging map/poll per field | Required by the current CPU-owned API; cross-frame changes are measurement and ownership gated. |
| RGB/YIQ conversion | CPU before/after the shared YIQ WGPU boundary | Format-independent effect parity; preview/export end-to-end cost must include it. |
| Device/pipeline setup and buffer allocation | Thread-local and amortized; frames reused until resize | Already avoids per-frame initialization/allocation. |

There is no hidden CPU image-effect fallback inside a selected WGPU stage. CPU work is control generation, coefficient construction, conversion, and orchestration. Explicit WGPU initialization failure resolves to CPU and now retains a concrete fallback reason.

## RNG and Stochastic-Effect Semantics

Current upstream is the resolved target. `SplitMix64::random` adds `0x9e3779b97f4a7c15`; u64 uses the standard SplitMix64 finalizer; u32/i32 and f32 use Stafford Mix4 with multipliers `0x62A9D9ED799705F5` and `0xCB24D0A5C88C35B3`; f32 uses the high 24 bits; `mix(input)` adds the input and performs one u64 finalization. Negative effect seeds first convert through u32, matching upstream.

The earlier count of seven was inconsistent. There are 10 stochastic processing paths, containing 11 named seed streams because head switching has a separate mid-line stream:

| Path | Seed/state construction | Dependence and location | Safe parallelization |
|---|---|---|---|
| Composite FBM noise | seed -> mix tag 0 -> mix frame -> clone/mix row; then i32 seed and f32 offset | CPU row controls; WGSL FBM pixels | Yes, rows independent; implemented. |
| Luma FBM noise | Same with tag 10 | CPU row controls; WGSL FBM pixels | Yes; implemented. |
| Chroma I FBM noise | Same with tag 1 | CPU row controls; WGSL FBM pixels | Yes; implemented. |
| Chroma Q FBM noise | Same with tag 9 | CPU row controls; WGSL FBM pixels, dispatched after I | Yes; implemented. |
| Chroma phase noise | tag 4 and frame, clone/mix row, one f32 | CPU sine/cosine per row; WGSL rotation | Yes; implemented. |
| Head switching | tag 2 and frame, clone/mix affected-row index | CPU row shifts; WGSL shift | Yes in principle. |
| Head mid-line jitter/transient | independent tag 8 and frame, sequentially draws two jitter f32 values and one transient f32 | CPU one-row control; WGSL partial shift/transient | Independent of other rows, preserve draw order. |
| Tracking | tag 3 and frame; base stream sequentially draws noise seed and offset; clone/mix row for row noise and row snow | CPU controls and ordered Snow merge; WGSL shift/noise/snow | Do not naively parallelize the shared merge. |
| Top-level snow | tag 6 and frame; clone/mix row; within-row geometric walk and independently seeded event pixel substream | CPU ordered event/tile merge; WGSL event evaluation | Do not naively parallelize the shared merge. |
| VHS edge wave | seed -> mix tag 5, then sequential i32 seed/f32 offset; no frame in RNG; frame enters 2D noise coordinate | CPU batched noise, WGSL shift | Batched already. |
| VHS chroma loss | tag 7 and frame, one sequential geometric walk accumulating row index | CPU lossy-row flags; WGSL zeroing | No. |

Deterministic phase error is not stochastic and was wrongly described as shader-side RNG. The removed `xoshiro.wgsl` seeded Xoshiro from standard SplitMix64, while its only included shader branch actually used an old Murmur finalizer and was never live because that dispatch set noise intensity to zero. Stochastic phase noise has always used `prepare::phase`. Removing that dead branch both eliminates obsolete semantics and reduces shader source parsing/compilation work.

## Target WGPU Architecture

Keep the existing planar Y/I/Q buffers, persistent backend, cached resources, single compute submission per field, and two-field overlap.

A cross-frame ring is `MEASURE_FIRST`, not an established next architecture. `apply_effect_to_yiq_with_backend_preference` receives a caller-owned mutable `YiqView` and returns only after mapped data is copied back. GUI render then performs RGB conversion and pushes the buffer into GStreamer. A useful ring therefore needs an asynchronous ownership-bearing API above this synchronous boundary, proof that encode/conversion work overlaps, a legal maximum in-flight count, and timing split into preparation/upload/compute/map-wait/copy/conversion/encode. Adding hidden frame retention under the current borrowing contract is rejected.

GPU-resident preview is also architecture-gated. Preview owns CPU GStreamer buffers: input is converted to CPU YIQ, WGPU output is read back, CPU YIQ is converted into a strided output buffer, and `eguisink` uploads/displays that buffer. The current display path cannot consume a `WgpuTexture` or external surface. A resident path requires explicit WGPU-to-GStreamer/egui texture interop, platform-specific Vulkan/Metal/DX12 ownership and synchronization, and likely a new preview sink. It is not a local conversion-shader change.

## Application-Path Coverage

| Path | Exact routing |
|---|---|
| GUI preview | `NtscFilterSettings { backend_preference: None }`; `process_gst_frame` resolves None to `BackendPreference::Auto`. |
| GUI render/export | `RenderSettings.backend_preference`, default `Auto`, is passed as `Some(...)` into the same frame processor. |
| CLI render | The real CLI exposes `--backend auto|cpu|wgpu|cuda` and passes it into render settings. |
| After Effects | Reads `NTSCRS_BACKEND`, default `Auto`; passes computed downsample scale factors when scale-with-video-size is active. |
| OpenFX | Reads `NTSCRS_BACKEND`, default `Auto`; passes the host proxy scale directly. |

All routes converge on the live shared helper. Its return reports the actual backend, but most application call sites currently do not present fallback detail to users. Direct runner tests are therefore the authoritative execution proof.

## Remaining Correctness and Parity Work

Completed source-justified work:

1. Proxy scale fixed in CPU and WGPU through one helper.
2. Current-upstream RNG and all stochastic call sites migrated.
3. Current-upstream simplex SIMD behavior ported; WGSL 1D gradient sign repaired.
4. Safe noise and phase row preparation parallelized in the configured pool with exact sequential-reference tests.
5. Fixed-seed fixtures added for every stochastic control path and the full pinned-upstream default stack.
6. WGSL parse/validation, proxy-scale WGPU, Alternating parity, adapter/fallback, and command-submission proof tests added.
7. Speculative out-of-bounds demodulation neighbor loads replaced with explicit edge guards.
8. Obsolete Xoshiro/Murmur shader source removed.

Still runtime-gated:

1. Run all seven ignored WGPU golden tests on a Vulkan, Metal, or DX12 adapter. The environment has none.
2. If any static WGSL claim fails that suite, fix the shader against pinned upstream rather than changing the 0.002 tolerance.
3. Surface fallback reports through UI/plugin logging if product requirements demand end-user diagnostics; the core runner now retains the reason.

## Optimization Classification

| Candidate | Classification | Current decision |
|---|---|---|
| Parallel `prepare::noise` and `prepare::phase` | `DO_NOW` | Implemented with `with_thread_pool` and `ZipChunks`; exact sequential-reference test passes. |
| Remove dead Xoshiro/Murmur shader concatenation | `DO_NOW` | Implemented; removes unused WGSL parsing/compilation and a correctness hazard. |
| Reuse device, pipelines, frames, staging buffers, control buffers and bind groups | `DO_NOW` | Already implemented by the fork and retained. |
| Parallel head controls | `MEASURE_FIRST` | Measure preparation share first; row count is small. |
| Local parallel snow/tracking event generation plus deterministic merge | `MEASURE_FIRST` | Measure CPU preparation and merge costs; preserve row/event order exactly. |
| Cross-frame export ring | `MEASURE_FIRST` | Requires stage timings and an asynchronous ownership API above the current sync boundary. |
| GPU-resident preview | `MEASURE_FIRST` plus architecture blocker | Requires display interop design and per-platform feasibility, not just a shader. |
| Filter coefficient/cache changes | `MEASURE_FIRST` | Record coefficient-build time, cache hits, and eviction under real settings changes. |
| Workgroup size changes or pass fusion | `MEASURE_FIRST` | Require adapter-attributed GPU timestamp and end-to-end measurements. |
| Per-pixel independent RNG, reordered event streams, FIR approximation, reduced-resolution effects | `REJECT` | Changes upstream sequences, correlations, filters, or visible output. |

## Implementation Phases

1. Source parity baseline: complete for the verified CPU semantics and controls, subject to tests below.
2. Runtime WGPU parity gate: blocked only by lack of adapter in this environment.
3. End-to-end instrumentation: next source change if hardware profiling shows the existing benchmark lacks upload/map/conversion/encode separation.
4. Architecture/tuning: only after those measurements. Cross-frame and preview proposals are no longer assumed.

## Verification and Golden-Test Contract

Coverage was read in full. The WGPU suite now has seven ignored tests:

- deterministic per-stage matrix across all demodulation modes, luma/chroma filters and filter types, sharpening, smear, ringing, phase error, delay signs, vertical blend, VHS speeds, 1x1/2x2/odd/non-workgroup dimensions, frames and scales;
- stochastic/default matrix covering composite, luma, chroma, phase, head, tracking, dense snow, VHS edge wave/tape/loss, seeds and non-unit scales;
- Both/Upper/Lower/InterleavedUpper/InterleavedLower with odd/even sizes, reuse and field timebase;
- explicit Alternating upper/lower frame parity;
- scale-disabled non-unit proxy regression;
- application WGPU selection;
- RGB u8 balloons fixture.

Pixel formats are converted outside `WgpuBackend`, whose API begins and ends at planar `YiqView`; RGB u8 is covered end to end. Additional packed formats belong to YIQ conversion testing, not shader-stage parity.

Validation executed in this implementation pass:

- `cargo test -p ntsc-rs --lib`: 37 passed.
- `cargo test -p ntsc-rs --lib --features gpu-wgpu`: 39 passed, 2 adapter-required ignored.
- `cargo test -p ntsc-rs --features gpu-wgpu`: all nonignored package tests passed; 9 adapter-required tests ignored across unit and integration targets.
- `cargo test -p ntsc-rs --features gpu-wgpu --test gpu_wgsl_validation`: 1 passed.
- `cargo test -p ntsc-rs --features gpu-wgpu --test backend_selection -- --nocapture`: 3 passed, including explicit no-adapter fallback reason.
- `cargo check -p ntsc-rs --all-targets --features gpu-wgpu`: passed.
- `cargo check --workspace --all-targets`: blocked in `glib-sys` because this environment has no `pkg-config`/GLib development setup; the failure occurred before GUI/plugin compilation.
- Direct pinned-upstream versus fork harness: bit-for-bit equality for clean, isolated stochastic, VHS, and full-default cases.
- The ignored WGPU suite was invoked and failed at adapter creation, before any shader dispatch. This is an environment blocker, not a parity pass.

## GPU-Execution Proof

`NtscEffectRunner` now exposes requested backend, active backend, last backend, fallback reason, adapter info, and WGPU effect-submission count. `WgpuBackend::submitted_effects` increments only after the effect command buffer is submitted, excluding readback-only submissions. Focused tests assert requested WGPU, active/last WGPU, absent fallback reason, nonempty adapter metadata, and an increased submission count, then compare downloaded pixels.

On this host, explicit WGPU produces active CPU plus `WGPU initialization failed`; the nonignored test proves that fallback is disclosed. No test in this run claims a WGPU dispatch occurred. A software CPU adapter may establish shader correctness but must be rejected for hardware-performance claims.

## Performance Benchmark Contract

No performance benchmark was run in this implementation pass, and no historical software-Vulkan number is used as evidence.

Use `cargo bench -p ntsc-rs --features gpu-wgpu --bench backend_profile` only with the exact adapter name, backend, device type, driver, OS, settings, input and resolution recorded. CPU WGPU adapters may validate correctness but cannot support acceleration claims.

Before implementing cross-frame ownership or shader tuning, measure:

- control preparation, Y/I/Q upload, command encoding, GPU timestamps per pass where supported, readback submission, `map_async` wait, mapped copy, YIQ/RGB conversion, and encoder/GStreamer handoff;
- end-to-end warm and steady-state latency and throughput at small, SD, 1080p and 4K sizes;
- default, deterministic-only and stochastic-heavy stacks;
- cache allocation/hit/eviction behavior and resolution changes.

Require repeated samples with variance. Workgroup sizing or pass fusion must improve adapter-attributed GPU and end-to-end measurements without changing the existing 0.002 parity tolerance. RX 6800 and Apple M2 are the named hardware validation targets.

## Platform and Adapter Considerations

The source supports WGPU's Vulkan, Metal and DX12 backends. This Linux environment has no enumerated WGPU adapter, including no software Vulkan ICD. Therefore shader execution, device limits, loss and performance were not exercised here. Software adapters must be labeled as such and excluded from hardware-performance claims.

## Edge Cases and Failure Modes

Current focused coverage includes 1x1, 2x2, odd dimensions, non-workgroup multiples, both chroma-delay signs, filter delay beyond width, field parity, interleaved odd/even heights, proxy scaling, dense stochastic controls, buffer reuse, and explicit adapter-init fallback.

Still adapter-dependent or missing: resize during a live runner, device loss, deliberately exceeded storage/buffer limits, pathological 8K snow buffers, and long-run cache churn. These are not reasons to weaken existing parity cases.

## Open Decisions

1. The RNG target is resolved: pinned current upstream SplitMix64/Mix4, with no compatibility exception.
2. Cross-frame export pipelining remains measurement and API-design gated.
3. GPU-resident preview remains blocked on GStreamer/egui/native-surface interop.
4. The unused `backend/` scaffolding remains low-priority cleanup because live routing and tests now identify the actual path unambiguously.

## Completeness Reconciliation

Completed in this pass:

- complete current-upstream `apply_effect_to_yiq_field` and helper flow read;
- all 22 core upstream commits and potentially semantic hunks classified;
- every live WGSL module inspected for equations, constants, indexing, boundaries, order, workgroup guards, dispatch shape and state inputs;
- canonical 10-path/11-stream stochastic inventory established;
- complete WGPU golden/backend suite read and extended without duplicating covered cases;
- GUI preview, GUI render/export, CLI, AE and OpenFX preference arguments traced;
- frame/staging ownership, enqueue/map/poll/copy lifetime and current two-field overlap traced;
- preview YIQ/RGB/GStreamer/eguisink ownership traced;
- Rust tests and adapter-independent shader validation executed.

Static review plus tests found and fixed five material parity, safety, or observability defects: proxy scale loss, obsolete RNG semantics, obsolete simplex SIMD behavior including the WGSL 1D gradient sign, speculative edge-lane demodulation reads, and missing fallback/submission observability. Runtime WGPU equivalence remains explicitly unclaimed until an adapter runs the ignored suite.

## Handoff Contract

- Upstream `88f2df9` is authoritative for all effect and RNG semantics.
- Source overrides this document if a later contradiction is found; repair the spec and implementation together.
- A selected WGPU backend may use CPU control preparation but must not execute image-effect stages through the CPU reference.
- Do not call a stage equivalent until the adapter-backed focused test passes, with requested/active/last backend, fallback, adapter, submission count, and pixels all asserted.
- Do not widen the 0.002 tolerance, reorder stochastic work, or replace recursive filters with approximations.
- Keep cross-frame, preview and shader-tuning work measurement-gated.
