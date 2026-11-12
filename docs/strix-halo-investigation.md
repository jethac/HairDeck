# Strix Halo optimization and hardware investigation

The user clarified on 2026-11-12 that success includes interesting things to do with the hardware, with reproducible
findings suitable for a blog, rather than latency optimization alone. Keep the full optimization/deployment goal and
add hardware exploration. Experiments stay separate from the working default until justified by measurements.

## Questions to investigate

- Can an independent preprocessing stage (for example matting) run on the XDNA2 NPU concurrently with GPU SAM3?
  First establish runtime availability, operator coverage and model conversion, then compare quality and end-to-end
  latency. Existing driver/device: amdxdna and /dev/accel/accel0. AMD's current Linux Ryzen AI docs support BF16 CNN and
  encoder models. The host's runtime installation is not yet established. Avoid equating advertised TOPS with benefit.
- How does shared LPDDR5X constrain concurrency? Compare isolated stages and overlapping CPU/GPU work, and record
  memory use, frequencies and latency. Unified physical memory does not by itself establish zero-copy tensor access.
- Why does restructuring small diffusion GEMMs help this APU? Keep original, eager optimized and compiled timings,
  dimensions and bandwidth reasoning. Existing measurements already reject forced clocks, HIP graphs and TunableOp.
- What changes improve responsiveness without changing inference quality? Image caching and video frame streaming
  are implemented. Separate cold startup, first-use compilation, new images, repeated images and multiple variations.

## Latest evidence and unfinished work

- Live demo serves HTTP 200 with image caching. Latest integrated commits: 86c1324 and a7a58b0 on origin/rocm.
- Remote working tree contains the deployed source edits and probes but its HEAD still needs reconciliation with origin.
- VRM export through the resident Blender worker: 5.45 s total, 4.21 s Blender, 0.87 s static preview, 0.27 s prep,
  0.10 s validation. Export passed its real VRM validator; Blender serialization accounts for 3.76 s.
- SAM3 compilation trials are documented in rocm.md, not enabled in the demo.
- The --bf16-linears probe failed: SAM3's decoder forward_ffn deliberately uses float32 input outside autocast, so
  converting every Linear to BF16 breaks dtype compatibility. No production model was changed. Next experiment can
  restrict conversion to the vision backbone or explicitly preserve FP32-only modules; do not silently cast the
  decoder's float32 path just to force the trial through.
- Preview streaming matched decoded video hashes when both encoders received exactly the same rendered RGB frames.
  Separate scene rendering runs differed; do not attribute that difference to encoding without isolating the source.

For each experiment record the hypothesis, exact commands/revisions, cold/warm state, actual hardware/runtime,
output-quality comparison, measurements and conclusion. Include failed and neutral results; do not fabricate a
positive result or extrapolate isolated-stage wins into end-to-end claims. This is source material, not a blog draft.
