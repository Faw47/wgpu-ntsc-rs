---
goal: Improve GPU performance and achieve logic parity with the CPU NTSC effects implementation
version: 1.0
date_created: 2026-03-28
status: Superseded
tags: [refactor, performance, feature, parity, gpu]
---

# Introduction

> Historical plan: the completion marks below were not backed by actual GPU parity or performance tests. They are retained as history, not current status. See [the current GPU audit](../docs/gpu-audit.md).

This implementation plan outlines the steps required to address performance bottlenecks in the `ntsc-rs` GPU backend and bring its behavior into full parity with the reference CPU implementation. The plan focuses on optimizing WGSL shader dispatch granularity, minimizing buffer copying overhead, ensuring non-blocking GPU readbacks, and implementing missing NTSC effects to ensure visual consistency between the CPU and GPU render paths.

## 1. Requirements & Constraints

- **REQ-001**: The GPU pipeline order must strictly match the CPU pipeline order defined in `ntsc.rs` (`NtscEffect::apply_effect_to_yiq_field`).
- **REQ-002**: Compute shaders processing pixels (e.g., noise, snow) must utilize per-pixel parallel dispatch to maximize GPU occupancy.
- **REQ-003**: GPU readback operations must be asynchronous to prevent CPU thread stalling.
- **REQ-004**: All visual effects present in the CPU backend must be available and functionally identical in the GPU backend.
- **PERF-001**: Minimize or eliminate intra-kernel buffer copying (e.g., row scratchpad arrays in `vhs_shifting.wgsl`).
- **PERF-002**: Minimize CPU-side dispatch and `copy_buffer_to_buffer` overhead by combining passes or utilizing a ping-pong buffer strategy.

## 2. Implementation Steps

### Implementation Phase 1: Structural Pipeline Fixes & Asynchronous Readback

- GOAL-001: Realign the execution order of the GPU pipeline to match the CPU and implement non-blocking readbacks to improve responsiveness.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Reorder the pipeline in `wgpu_backend.rs` so that `luma_filter` and `chroma_lowpass_in` are applied *before* `chroma_into_luma`. | ✅ | 2026-03-28 |
| TASK-002 | Refactor `WgpuFrame::download` to use asynchronous buffer mapping (`map_async`) instead of synchronous `device.poll(Maintain::Poll)`. | ✅ | 2026-03-28 |
| TASK-003 | Update `wgpu_backend.rs` control logic to select and dispatch advanced `luma_into_chroma` demodulation filters (Notch, 1-line comb, 2-line comb) based on `RenderSettings`, replacing the hardcoded `Box` filter dispatch. | ✅ | 2026-03-28 |

### Implementation Phase 2: Compute Shader Optimizations

- GOAL-002: Resolve critical performance bottlenecks within WGSL shaders by utilizing correct dispatch granularity and avoiding costly internal operations.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-004 | Refactor `plane_noise.wgsl` to use per-pixel execution (`GlobalInvocationId.xy`) instead of per-row loops. | ✅ | 2026-03-28 |
| TASK-005 | Refactor `snow.wgsl` to use per-pixel execution (`GlobalInvocationId.xy`) instead of per-row loops. | ✅ | 2026-03-28 |
| TASK-006 | Rewrite `vhs_shifting.wgsl` to eliminate the $O(width)$ internal row copy to scratch memory. Implement this either by splitting into read/write passes or using workgroup shared memory properly. | ✅ | 2026-03-28 |
| TASK-007 | Implement a ping-pong buffer strategy in `wgpu_backend.rs` for sequential effect passes to eliminate unnecessary `copy_buffer_to_buffer` calls between intermediate stages. | ✅ | 2026-03-28 |

### Implementation Phase 3: Missing Effects Implementation

- GOAL-003: Port missing NTSC visual effects from the CPU logic to the GPU pipeline.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-008 | Implement `ringing` and `composite_sharpening` effects in the GPU pipeline (likely reusing or extending `filter_plane.wgsl`). | ✅ | 2026-03-28 |
| TASK-009 | Implement `luma_smear` effect logic for the GPU pipeline. | ✅ | 2026-03-28 |
| TASK-010 | Implement `chroma_phase_error` and `chroma_phase_noise` effects in WGSL. | ✅ | 2026-03-28 |
| TASK-011 | Implement specialized VHS tape speed lowpass filters for the GPU backend. | ✅ | 2026-03-28 |

### Implementation Phase 4: Advanced Logic Parity

- GOAL-004: Port complex CPU-side logic for specific effects that are currently oversimplified on the GPU.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-012 | Analyze `ntsc.rs` for the complex `mid_line` jitter and transient pulse logic associated with head switching. | ✅ | 2026-03-28 |
| TASK-013 | Implement the `mid_line` jitter and transient pulse head switching logic within `vhs_shifting.wgsl` (or a dedicated shader pass) to achieve exact visual parity with the CPU tape tear effect. | ✅ | 2026-03-28 |

## 3. Alternatives

- **ALT-001**: Instead of ping-pong buffers, combine multiple compatible shader passes into "mega-passes" (e.g., doing phase shifts, noise, and chroma/luma mixing simultaneously). *Reason not chosen initially: High complexity and potential register pressure; ping-pong buffers provide an easier initial win for eliminating explicit copies.*
- **ALT-002**: Leave readback synchronous but run the GPU dispatch on a dedicated background thread. *Reason not chosen initially: Asynchronous readback via mapped buffers is the idiomatic WebGPU/wgpu approach and avoids thread synchronization overhead.*

## 4. Dependencies

- **DEP-001**: `wgpu` crate (Requires ensuring the asynchronous buffer mapping logic complies with the current `wgpu` version's API).
- **DEP-002**: `pollster` or similar async executor (May be needed if the application doesn't currently have an async runtime to await the buffer map futures).

## 5. Files

- **FILE-001**: `crates/ntscrs/src/gpu/wgpu_backend.rs` (Main pipeline orchestrator, needs extensive updates for execution order, dispatch logic, and async readback).
- **FILE-002**: `crates/ntscrs/src/gpu/shaders/plane_noise.wgsl` (Needs per-pixel refactor).
- **FILE-003**: `crates/ntscrs/src/gpu/shaders/snow.wgsl` (Needs per-pixel refactor).
- **FILE-004**: `crates/ntscrs/src/gpu/shaders/vhs_shifting.wgsl` (Needs internal copy optimization and complex head-switching logic porting).
- **FILE-005**: `crates/ntscrs/src/ntsc.rs` (Reference material for execution order and effect logic).
- **FILE-006**: `crates/ntscrs/src/gpu/shaders/filter_plane.wgsl` (Will likely be modified or utilized for missing IIR effects).

## 6. Testing

- **TEST-001**: Extend `crates/ntscrs/src/gpu/tests.rs` or `gpu_wgpu_golden.rs` to visually compare the output of the CPU and GPU backends for the newly implemented effects (`ringing`, `sharpening`, etc.) ensuring strict visual parity.
- **TEST-002**: Benchmark the GPU pipeline before and after the WGSL dispatch refactors (Tasks 004-006) to quantify the performance improvements.

## 7. Risks & Assumptions

- **RISK-001**: Porting complex CPU logic like head-switching transient pulses to WGSL might introduce precision issues or require significant refactoring of how random noise state is passed to the shader.
- **ASSUMPTION-001**: The wgpu device and queue are initialized with limits that support the required compute workgroup sizes for per-pixel dispatch without exceeding hardware maximums.

## 8. Related Specifications / Further Reading

- [wgpu Map Buffer Documentation](https://docs.rs/wgpu/latest/wgpu/struct.Buffer.html#method.map_async)
- [WGSL Specification - Compute Shaders](https://www.w3.org/TR/WGSL/#compute-shader)