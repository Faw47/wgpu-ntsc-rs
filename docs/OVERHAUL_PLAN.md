# GPU and desktop overhaul

Baseline: audited `517538a8367182015b88c020a510d0c473335d01`.
Semantic authority: upstream `88f2df9a27863097eaffd8d1fe0080a174dfd4a5`.

## Delivery ledger

1. Integrate the audited portability repairs into main, preserving exact arithmetic and strict explicit-WGPU failures.
2. Add opt-in timestamp queries around every compute dispatch. Report adapter identity and distinguish software adapters. Include CPU preparation/upload/compute/readback in end-to-end measurements separately from timestamps.
3. Implement a CPU block-state formulation for a representative low-order IIR, validate against the reference, then implement a separate WGPU experiment and parity-gated measurement command. Do not enable it in normal rendering without broad parity and physical-GPU speed evidence.
4. Replace the standalone workspace layout: application command bar, dominant preview, right inspector with Effects/Presets/Export, searchable parameter groups, usable empty state and recent media, accessible transport and comparison controls, export queue and persistent error feedback. Preserve all settings, preset formats, media operations and codecs.
5. Build and visually inspect the real desktop app at laptop and narrow window sizes; exercise media load, settings/search, comparison, presets, export and errors. Run the GPU acceptance tests.
6. Merge verified code into main with no outstanding PR for this work. Record hardware and numerical limits honestly.
7. Performance follow-up: batch interlaced readbacks, expose bounded asynchronous WGPU submission, reuse YIQ/control allocations, negotiate export precision, and add strict-versus-fast arithmetic plus filter-workgroup tuning knobs.

## Design contract

Mode: redesign/overhaul. Rust + egui/eframe 0.33.3 and GStreamer remain the implementation stack.
Audience: creators processing real footage. Primary job: open media, adjust analog degradation, compare, export.
Visual family: neutral editing bench. Signature: source/processed comparison in the preview, with a small restrained signal-color mark.
Dials: variance 5, motion 2, density 7. No continuous chrome animation.
Dark roles: canvas #151719, panel #202326, control #2C3034, text #E8EAEC, muted #A9B0B7, selection #8DBCB3. Light equivalents retain contrast and neutral preview surround.
Type: bundled proportional egui face for controls, monospace for time/data. 13 px body, 11 px secondary, 18 px workspace headings. No new font dependency.
Spacing: 4/8/12/16/24. Bounded corner radius 4. Flat panels and single separators; no nested glass cards.
Layout: top commands, right resizable inspector, preview fills remainder. Inspector width capped to retain preview; focus mode hides inspector. Controls wrap at narrow widths and scrolling retains access to all parameters.
States: keyboard focus from egui, disabled controls for unavailable media operations, explicit search-empty feedback, clearable search, retained errors with dismissal, real queue state. No invented GPU performance indicators.
Protected behavior: settings IDs and ranges, presets and undo/redo, dialogs and legal attribution, CPU/GPU processing contract, timeline and split preview, codec settings and output validation.
Database suggestions for landing pages, pink branding, GSAP and marketing statistics were rejected as mismatched. The local design database has no egui stack profile; installed crate source is the API authority.

## GPU method and gates

Block-state propagation follows affine composition and boundary-state decomposition, as documented by the GPU recursive filtering literature linked at https://github.com/andmax/gpufilter. No claim from the unavailable July 2026 arXiv link is relied upon.
A block computes its zero-state final state; a prefix propagation computes each block's incoming state; independent blocks replay from those states into a separate output. Coefficients, initial conditions, delay and edge extension are retained. Floating-point reassociation is explicitly an experimental numerical change.
The current implementation completes the safe host-side and submission-side follow-up: progressive effect and readback work share one command buffer, interlaced readback copies share one command buffer, `WgpuBackend::apply_effect_async` supports bounded callers, immutable device and pipeline state is shared across runners, independent control streams are prepared concurrently with a parity check, YIQ storage is reused, 8-bit exports avoid unnecessary Argb64 output, strict/fast arithmetic plus filter-workgroup variants are validated at the WGSL level, and an opt-in progressive `Both` preview can convert YIQ to packed RGBA8 on the adapter while reusing scratch/staging storage. Physical RX 6800/Metal/DX12 performance and the decision to enable the experimental block-IIR path are outside what a software Vulkan runner can establish. Full GPU residency in eframe and a multi-frame GStreamer export pipeline remain contingent on sharing the renderer device and changing the synchronous transform contract.
