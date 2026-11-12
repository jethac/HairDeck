# Strix Halo optimization and hardware investigation

The user clarified on 2026-11-12 that success includes interesting things to do with the hardware, with reproducible
findings suitable for a blog, rather than latency optimization alone. Keep the full optimization/deployment goal and
add hardware exploration. Experiments stay separate from the working default until justified by measurements.

## Questions to investigate

- Can an independent preprocessing stage (for example matting) run on the XDNA2 NPU concurrently with GPU SAM3?
  First establish runtime availability, operator coverage and model conversion, then compare quality and end-to-end
  latency. Existing driver/device: amdxdna and /dev/accel/accel0. AMD's current Linux Ryzen AI docs support BF16 CNN and
  encoder models. Native Arch runtime and vendor quicktest work; the HairDeck BF16 probe now has a verified
  single-stamp compiler workaround (see below), but remains slower than the GPU.
  Avoid equating advertised TOPS with benefit.
- Does overlapping actual generation and preview work help on this APU? The experiment below measures this directly.
  Unified physical memory does not by itself establish zero-copy tensor access or memory-bandwidth saturation.
- Why does restructuring small diffusion GEMMs help this APU? Keep original, eager optimized and compiled timings,
  dimensions and bandwidth reasoning. Existing measurements already reject forced clocks, HIP graphs and TunableOp.
- What changes improve responsiveness without changing inference quality? Image caching and video frame streaming
  are implemented. Separate cold startup, first-use compilation, new images, repeated images and multiple variations.

## Deployment and measured decisions

- Live demo serves HTTP 200 with image caching and streamed preview encoding. It now runs as a foreground Compose
  service with `restart: unless-stopped`; Docker is enabled at boot. Real API acceptance generated two variations,
  exported a validated 13.7 MB VRM, and rendered a three-second motion preview. The old development container is
  stopped and retained. Acceptance ran during a source build, so its timing is not a clean performance measurement.
- VRM export through the resident Blender worker: 5.45 s total, 4.21 s Blender, 0.87 s static preview, 0.27 s prep,
  0.10 s validation. Export passed its real VRM validator; Blender serialization accounts for 3.76 s.
- SAM3 compilation trials are documented in rocm.md, not enabled in the demo.
- The --bf16-linears probe failed: SAM3's decoder forward_ffn deliberately uses float32 input outside autocast, so
  converting every Linear to BF16 breaks dtype compatibility. No production model was changed. The probe also
  supports `--bf16-vision` for limiting conversion to the backbone; full-model conversion is rejected.
- Preview streaming matched decoded video hashes when both encoders received exactly the same rendered RGB frames.
  Separate scene rendering runs differed; do not attribute that difference to encoding without isolating the source.

For each experiment record the hypothesis, exact commands/revisions, cold/warm state, actual hardware/runtime,
output-quality comparison, measurements and conclusion. Include failed and neutral results; do not fabricate a
positive result or extrapolate isolated-stage wins into end-to-end claims. This is source material, not a blog draft.

## Initial HairDeck MLP NPU experiment: incorrect output

`tools/rocm/bench_npu_mlp.py` exports actual `blocks.0.mlp` weights from the released image model. Inputs are synthetic
standard-normal activations, seed 42, shape (2,192,768); the MLP expands to 3072 channels and uses tanh GELU.
This is an isolated hardware/compiler probe, not an end-to-end HairDeck benchmark or quality acceptance.

On native Arch/XRT 2.27/Ryzen AI 1.9, the BF16 VitisAI compiler placed the graph on the NPU. Median MLP inference was
8.31 ms with 0.813 relative L2 error, versus the ROCm GPU's 0.239 ms compiled and 0.00226 relative L2 against CPU FP32.
The GPU measurement includes synchronization but resident inputs; the NPU ORT run includes its host interface. The
paths are not comparable as raw kernel throughput, and the NPU result is numerically invalid regardless of speed.

CPU ONNX matches CPU PyTorch (relative L2 1.69e-7), ruling out the export as the source of the large error. Flattening
the input to (384,768) preserves the same NPU error. Isolating just the first Linear (`--linear --flat`) gives 1.41 ms
and 0.708 relative L2 error. Exactly half the output channels are zero: columns 0..1535 have roughly 0.0094 relative
L2 error, while columns 1536..3071 are all zero. NPU placement was confirmed in the compiler log. Root cause remains
open: investigate output-channel tiling and the newer XRT/runtime combination before attributing this to hardware.
The probe saves reference tensors, ONNX, output tensors and metrics in its specified --root for a reproducer.

Native Python's executable is copied into the venv rather than symlinked; locating libpython via readlink was wrong.
The Arch fork's run-sdk.sh now asks sysconfig for LIBDIR. This resolved the BF16 runtime's missing-libpython error.

### Matched runtime comparison

Reducing the isolated Linear to 768 output channels (`--linear --flat --width 768`) still gives relative L2 0.70665.
AMD xdna-driver branch 1.9 at `a1f7a2fe`, with pinned XRT `e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6` (2.26.0),
built natively with `SKIP_KMOD=ON` and a private install prefix. Recompiling with fresh caches on that stack gives
the same error with both `--layout auto` and `--layout vectorized`. `/proc/self/maps` confirmed that core, coreutil
and XDNA shim libraries all loaded from the private 2.26 directory. The newer 2.27 userspace version and the layout
choice therefore do not explain this failure. No kernel driver was replaced. The defect was subsequently localized
and mitigated as described below. The working ROCm path is unaffected.

## Concurrent preview and generation

`python tools/rocm/bench_overlap.py` measures four cached-image variations and a separate 90-frame motion preview,
first alone and then together. The preview runs CPU spring simulation, GPU OpenGL rendering and CPU ffmpeg encoding;
generation runs the real ROCm models, colors, chains and GLB output. Fresh output directories bypass preview caching.

In the recorded pair, generation alone took 2.349 s and preview alone 2.969 s (5.318 s combined serial time).
Overlapping them completed both in 2.963 s; generation took 2.472 s and preview 2.961 s. This shows useful overlap
for this workload with a small generation latency increase. It does not establish raw LPDDR5X bandwidth, general
scaling, or NPU/GPU overlap. One pair is a diagnostic, not a statistical throughput claim. The demo already permits
independent preview and generation jobs, so no additional scheduling change was needed. Raw results are retained in
[strix-halo-overlap.json](measurements/strix-halo-overlap.json).

The deployed workflow's process-cold, new-image, repeat-image and multi-variation timings are in
[rocm.md](rocm.md#deployed-workflow-measurements), with a runnable probe and raw measurements.

## BF16 diagnosis and verified workaround (2026-11-13)

The failure is in the compiler's work partitioning and generated graph. A checkpoint-free identity Gemm with
M=128, K=N=768 reproduces it exactly. XRT buffer tracing confirms the input reaches the device intact and the returned
BF16 output already has zeroes in its second half, before host conversion. Compiler intermediates plan two compute
stamps split across output channels, but the generated graph instantiates only stamp 0, covering 384 of 768 channels.

Setting `vaiml_config.tp_size=1` makes the tiler cover all output channels with the emitted stamp. The identity probe
then matches the CPU exactly. The original released-weight Linear and three-dimensional MLP produce no all-zero
channels, with relative L2 0.00944 and 0.00807 respectively against FP32. The compiler uses BFP16 weight packing,
so the residual error should not be attributed solely to BF16 activation rounding.

`tools/rocm/bench_npu_mlp.py` now defaults to this workaround, separates caches by tp_size, and fails if relative L2
exceeds the probe's explicit 2% bound. `--tp-size 0` retains the automatic path for reproducing the defect. This is
a scoped compiler configuration workaround, not an upstream compiler patch or end-to-end NPU HairDeck acceptance.
The corrected MLP remains slower than ROCm, so no NPU path is enabled in the demo. Complete commands, trace source,
library hashes, generated-code evidence and limitations are in the
[Arch fork diagnosis](https://github.com/jethac/ryzen-ai-arch/blob/main/docs/bf16-stamp-mismatch.md).
