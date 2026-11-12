# Continued Strix Halo optimization after the NPU workaround

The NPU's corrected BF16 MLP is still slower than the compiled GPU MLP. Working arithmetic alone does not
justify inserting it into the GPU sampler. This pass measures batching and repeated GPU work, and probes
whether the existing independent matting stage can reach the NPU compiler.

**Measurement boundary correction:** the deployed `best` preset in `hairdeck/prep/presets.json` uses
BiRefNet at **768px**, although SAM's working resolution is 1024px. The earlier standalone matting
benchmarks and export/compilation diagnostics below used **1024px**. They are not measurements of the
deployed matting configuration. The production-size experiment at the end uses 768px and the actual
alpha-compositing rule. Matting still runs for transparent inputs; the input alpha limits its result
afterward rather than bypassing the network.

Reference-precision convention: these export probes load the production wrapper's FP16 weights and
widen them for FP32 arithmetic before comparing reformulations. "FP32 reference" below means that
unchanged pre-reformulation model, not a reload of the unrounded FP32 checkpoint. The production-size
probe additionally retains the original FP16 logits and FP16 sigmoid output for a direct comparison.

## Optional batched image variations

`Pipeline.run(..., batch_variations=True)` batches the image DiT across independently seeded noise tensors.
VAE decoding, colors, chains and file export retain their per-seed paths. Single-variation requests, model
offload and ZeroGPU keep sequential sampling. `HAIRDECK_BATCH_VARIATIONS=1` enables it for the demo; the Compose
file forwards this variable. It defaults to **off**, because batch shape changes floating-point rounding
and generated output. Set the environment variable before creating/recreating the service, not just in an
unrelated shell after it is running.

Run `python tools/rocm/bench_variations.py` inside the ROCm environment. This uses example 2, seeds 1 through
the requested count, 50 steps, CFG 2, colors, chains and GLB files. The live demo was resident but idle.
These are individual request observations, not medians or percentiles. Disk compiler caches were retained.

| Warm cached-image request | Sequential total | Batched total | Sequential sampling + decode | Batched sampling + decode |
|---|---:|---:|---:|---:|
| 2 variations | 1.173 s | 1.046 s | 0.648 s | 0.524 s |
| 4 variations | 2.280 s | 1.968 s | 1.326 s | 1.017 s |

The corresponding complete-request reductions are 10.8% and 13.7%. The first batch-2 request took 7.090 s,
and the first batch-4 request 8.428 s, including shape-specific compilation/initialization. These are not
isolated compiler times or empty-cache measurements. The initial sequential request took 8.244 s, including
uncached preprocessing and other first-use costs; it is not a matched cold comparison.

Warm peak allocated GPU tensor memory was 7.262 GiB for both two-variation paths, and 7.264 versus 7.411 GiB
for four variations. Reserved memory is reported separately in the raw artifact and depends on preceding
requests. These measurements exclude the separate demo process and are not total machine memory.

Same-seed raw tokens were **not identical**. Relative L2 differences ranged from 0.00667 to 0.01003 at batch 2,
and 0.00348 to 0.03766 at batch 4. Card counts stayed the same in these samples, but seed 1 changed from 17
chains to 16 at batch 4. The seed-3 first preview frames retained the broad hairstyle with visible local
strand/shape differences. This limited visual inspection is not a quality-equivalence evaluation.

Batch-4 seed 3 also completed VRM export and validation: 116 cards, 18 chains, 13.9 MB VRM, no reported
validation problems. The 90-frame motion preview completed. Artifacts are retained on the target under
`~/hairdeck-root/results/variation-benchmark/`; the reference and batched seed-3 videos permit comparison
throughout the motion rather than just the first frame.

Raw results: [request timings](measurements/strix-halo-variations.json),
[batch-2 numerics](measurements/strix-halo-variations-2-numerics.json),
[batch-4 numerics](measurements/strix-halo-variations-4-numerics.json).

## Repeated conditioning: measured, additional cache deferred

`python tools/rocm/profile_conditioning.py` loads the real pipeline and the saved example-2 crop, warms each
operation three times and records 20 synchronized wall-clock observations. This diagnostic measures the
operations independently; its synchronization is not inserted into production requests.

| Operation | Median |
|---|---:|
| Existing cached DINO feature lookup, including pixel hash | 0.271 ms |
| Image-model condition projection | 0.196 ms |
| Fast denoiser construction, including fixed KV and all-step modulation | 4.792 ms |

Even eliminating these costs entirely would save only about 5.3 ms per sequential variation in this probe.
A persistent denoiser cache would retain sizable tensors and need invalidation for image, step count,
guidance, weights and device. We did not add it for this small measured opportunity. Batching already
shares condition preparation within a request. [Raw samples](measurements/strix-halo-conditioning.json).

## Whole-stage NPU: matting export obstacle

BiRefNet is an independent preprocessing stage and a reasonable candidate for overlapping an earlier
image's GPU generation. `python tools/rocm/export_npu_matte.py` attempted the actual 1024px model export,
without replacing its operators. PyTorch's legacy ONNX exporter at opset 17 rejected
`torchvision::deform_conv2d` with `UnsupportedOperatorError` after 2.32 s.

This is a **frontend export failure**, before NPU partitioning or compilation. It does not establish
hardware impossibility, NPU latency, or CPU fallback performance. Opset 17 was chosen because AMD's
[deployment documentation](https://ryzenai.docs.amd.com/en/latest/modelrun.html) recommends it.
The [result](measurements/strix-halo-npu-matte-export.json) records the exact error and Torch version;
the target retains the traceback under `~/hairdeck-root/results/npu-matte-export/`.

### Export obstacle resolved with a fixed-shape decomposition

The subsequent experimental exporter replaces the single-group deformable convolution with nine
bilinear `GridSample` operations for a 3x3 kernel, the original modulation mask, and a 1x1 convolution
using the reshaped learned weights. Zero padding and offset ordering are retained. This adapter lives
only in `tools/rocm/deform_grid.py`; the production model still uses torchvision's HIP operator.
AMD's [operator table](https://ryzenai.docs.amd.com/en/latest/ops_support.html) lists BF16 GridSample,
but that is a reason to try compilation, not proof that this graph maps to the NPU.

On example 2 at the real 1024px resolution, the full FP32 model with this decomposition differed from
the original FP32 model by relative L2 **2.103e-7**, maximum logit difference **6.008e-5**, mean alpha
difference **4.041e-9**, and maximum alpha difference **7.004e-6**. This is one image, not a dataset quality
claim or a comparison against the deployed FP16 path. See [equivalence metrics](measurements/strix-halo-matte-grid-equivalence.json).

The legacy exporter initially could not infer the reshaped convolution kernel. Making dimensions
concrete for this deliberately fixed-shape export resolved that failure. The full ONNX export then
succeeded in 9.377 s and produced an approximately 890 MiB model. Export time excludes model loading,
the reference/decomposition comparison and its first-use convolution searches.
[Export result](measurements/strix-halo-matte-grid-export.json).

The exported ONNX graph also ran through native ONNX Runtime 1.30 CPU execution against the saved
PyTorch FP32 reference. Relative L2 was **2.178e-6**, mean alpha difference **1.445e-8**, maximum alpha
difference **3.940e-5**, and threshold-0.5 mask IoU **1.0** on this image. This independently validates
the exported graph before judging the NPU result. [CPU export comparison](measurements/strix-halo-matte-grid-onnx-cpu.json).
Pass `--cpu-reference` to the native stage command below to reproduce it; it does not benchmark latency.

Reproduce the export in the GPU container:

```bash
python tools/rocm/export_npu_matte.py --decompose
```

The native Arch stage probe uses the single-stamp workaround, records runtime profiling/placement,
and checks output before collecting timings. Its numerical gates are diagnostic acceptance limits,
not an assertion that any output below them has equal downstream hair quality:

```bash
bash ~/dev/ryzen-ai-arch/scripts/run-sdk.sh \
  ~/hairdeck-npu/native/runtime/opt/amd/ryzenai/1.9.0 \
  ~/hairdeck-npu/native/venv/bin/python \
  ~/dev/HairDeck/tools/rocm/run_npu_stage.py \
  --root ~/hairdeck-root/results/npu-matte-grid
```

Next research steps are inspection of actual NPU partitions, numerical comparison and complete-stage timing. A smaller replacement matte
or segmentation model is another route, but would require suitable weights and a quality comparison on
real inputs; none was substituted here. Whole-stage NPU latency, GPU/NPU overlap, transfer cost, contention
and power remain unmeasured. The working native Arch runtime is preserved and production uses ROCm.

### Compact equivalent export

The first native compile spent over 18 minutes in tiling without reaching execution. A single debugger
snapshot showed `tilingFeasibilityCheck::checkFeasibility` and local-memory metadata calculations; its
intermediate artifacts continued changing. It was left running, not classified as a failed compile.

`python tools/rocm/export_npu_matte.py --compact-grid` packs all kernel sampling positions into one
GridSample output per deformable convolution, then restores the original channel/kernel ordering before
applying the learned weights. This keeps the model and sampling rule intact while reducing graph size.
The full-model FP32 and exported ONNX CPU comparisons produced the same error metrics as the earlier
decomposition on example 2. [PyTorch comparison](measurements/strix-halo-matte-compact-equivalence.json),
[ONNX CPU comparison](measurements/strix-halo-matte-compact-onnx-cpu.json).

Raw exported nodes decreased from 31,732 to 21,597, and GridSample nodes from 300 to 20. After the runtime's
initial optimization, the compiler log reported 3,198 nodes instead of 8,458. These are graph counts, not
latency improvements. [Raw operator counts](measurements/strix-halo-matte-compact-graph.json).
The compact export took 7.210 s in this observation, excluding loading and equivalence checks;
[export metadata](measurements/strix-halo-matte-compact-export.json).

The compact model has its own `npu-matte-grid-compact` output/cache directory. Its native probe was started
with `--validate-only` while the original compiler remained running; no uncontended latency measurement
can be inferred from this overlapping compilation experiment. This compact compile later failed as described below.

### Focused real-activation layer diagnostic

While both complete-model compiles ran, `python tools/rocm/export_matte_deform.py` captured the first
deformable convolution from example 2's actual BiRefNet forward pass. This is a 1x1 learned convolution
with nontrivial offsets/modulation, input `[1,64,32,32]` and output `[1,256,32,32]`, not a whole-stage result.
The compact FP32 decomposition differed from torchvision by relative L2 **2.514e-7**.

The native `run_npu_stage.py --feature-output --validate-only --root .../npu-matte-deform` probe compiled
and executed it. The compiler log reported **25 VAIML operators, one NPU subgraph, one actually running
on NPU**, and the runtime profile recorded one VitisAI execution event. Session creation took 11.616 s
while the larger compiles were running; this is not an uncontended compilation benchmark.

The layer output's relative L2 error was **0.023357**, above the probe's predeclared 0.02 gate. No latency
benchmark was collected. This does not establish a whole-matte quality failure or another compiler
defect: numerical effects must be evaluated on the final alpha output. The raw report's old error text
says "Whole-stage numerical gate"; its `feature_output: true` and this captured shape identify it as a
layer probe. The error text was subsequently made generic.
[Captured-layer metadata](measurements/strix-halo-matte-deform-export.json),
[native result](measurements/strix-halo-matte-deform-npu.json).

### Coordinate formulation improves the captured NPU layer

`export_matte_deform.py --stable-coordinates` normalizes the constant base grid first, then adds the
scaled offset: `base_normalized + offset * (2 / size)`. The earlier expression added the small offset
to a potentially large pixel coordinate before normalizing. The two expressions are algebraically
equivalent, but differ under reduced precision.

The revised FP32 layer differed from torchvision by relative L2 **6.433e-7**. On the NPU, relative L2
against the same original FP32 layer was **0.015115**, below the 0.02 diagnostic gate. The log recorded
19 VAIML operators and one subgraph actually running on NPU. This is evidence that formulation matters
for this captured layer; it is not proof that every remaining error comes from coordinate rounding,
nor a whole-matte quality result. No timing was collected during concurrent full-model compilation.
[Export metadata](measurements/strix-halo-matte-deform-stable-export.json),
[native result](measurements/strix-halo-matte-deform-stable-npu.json).

### Full-model compiler buffer failure, reduced to real attention

The compact full-model compile terminated with AIE compiler error **77-6416**: an attention-transpose
buffer dimension of **276480** exceeded its allocated **262144** elements. There was no whole-model
execution or latency result. The native compiler terminated the process directly, so the Python probe
could not write its usual exception report; the compiler log and process exit code are the evidence.

`python tools/rocm/export_matte_attention.py` captures the actual second-scale
`bb.layers.1.blocks.0.attn` module, with real example-2 input `[36,144,384]`. The exported module reproduces
the same error at `/attention/Transpose_1`, in a much smaller graph. Forcing `--layout unvectorized`
also failed with the identical buffer mismatch. Isolating only Transpose, or Transpose plus a dynamic
MatMul, did **not** reproduce it: those graphs executed on CPU and the probe correctly rejected them as
NPU evidence. Retaining the real attention module and weights preserves the failing compiler path.

[Captured attention metadata](measurements/strix-halo-matte-attention-export.json),
[compiler errors and exit codes](measurements/strix-halo-matte-buffer-failures.json).
The original, larger decomposition compile was a separate experiment; do not treat the compact model's
failure as its result. It was subsequently cancelled as superseded by the validated attention-layout
workaround, after 50:32 elapsed, preserving its artifacts and recording no execution result.
[Cancellation record](measurements/strix-halo-matte-original-cancelled.json).

### Equivalent attention layout compiles on the NPU

`export_matte_attention.py --flat-heads` folds windows and heads into one batched-matmul dimension.
It preserves all windows, heads, learned weights, relative position bias and masks. On the captured
`[36,144,384]` attention input, the FP32 result was identical to the original module (relative L2 zero).
The NPU compile then succeeded: 19 VAIML operators, one subgraph actually running on NPU, and relative
L2 **0.011146** against FP32, below the diagnostic 0.02 gate. Session creation was 8.778 s with the original
large compilation still running, so it is not an uncontended performance benchmark.
[Export metadata](measurements/strix-halo-matte-attention-flat-export.json),
[native validation](measurements/strix-halo-matte-attention-flat-npu.json).

This avoids the buffer mismatch by changing the compiler-visible layout without changing the model's
attention operation. It is an export workaround, not a patch to AMD's tiler. It does not establish that
all shapes or an entire model now compile correctly.

The combined full-model export applies this attention layout to every Swin attention module, and uses
compact sampling with the improved coordinate formulation:

```bash
python tools/rocm/export_npu_matte.py --flat-heads --stable-coordinates
```

Against the unchanged full FP32 BiRefNet on example 2, relative L2 was **3.623e-7**, mean alpha difference
**1.034e-8**, and maximum alpha difference **1.013e-5**. It exported successfully and passed the separate
ONNX CPU check before a new native NPU compilation was started with `--validate-only`.
[Full-model equivalence](measurements/strix-halo-matte-reformulated-equivalence.json),
[export metadata](measurements/strix-halo-matte-reformulated-export.json).
Its artifacts live under `npu-matte-grid-compact-flat-stable`; that particular native attempt subsequently
failed for the constant-expansion reason below, before execution.

### Preserve broadcasting to avoid a compiler serialization limit

The first full flat-head export was approximately 887 MiB, but the native compiler's internal ONNX
model grew to **3,579,387,916 bytes**, exceeding protobuf's 2 GiB model-serialization limit. Explicitly
expanding relative-position bias across all windows before flattening allowed the compiler to materialize
large repeated constants. [Native error](measurements/strix-halo-matte-expanded-bias-failure.json).

The adapter now adds the bias while scores have their original `[windows,heads,tokens,tokens]` shape,
using broadcasting, and then flattens scores again. This retains the three-dimensional attention
matrix products that avoided the tiling failure, without repeating the bias tensor. The captured
attention module still compiles and passes NPU validation with relative L2 **0.011146**.
[Native reduced-module result](measurements/strix-halo-matte-attention-broadcast-npu.json).

The combined model again passed its original FP32 and ONNX CPU comparisons, and the current native
attempt uses `npu-matte-grid-compact-flat-stable-broadcast`. The old long compile was stopped before
this attempt, so it will not contaminate inference timings if the numerical gate passes.
[FP32 equivalence](measurements/strix-halo-matte-broadcast-equivalence.json),
[export metadata](measurements/strix-halo-matte-broadcast-export.json),
[ONNX CPU comparison](measurements/strix-halo-matte-broadcast-onnx-cpu.json).
That 1024px diagnostic was subsequently cancelled after the deployed resolution was confirmed, retaining
its artifacts without an execution result. [Cancellation record](measurements/strix-halo-matte-broadcast-cancelled.json).

### Match the deployed 768px matting stage

```bash
python tools/rocm/export_npu_matte.py --flat-heads --stable-coordinates --size 768 --pipeline-input
```

This uses the complete same BiRefNet model, at the resolution already configured by the live `best`
preset. It is not a smaller replacement model or a newly reduced-quality preset. `--pipeline-input`
uses the application's handling of transparent pixels before resizing and normalization.
The original FP16 network's sigmoid alpha is saved alongside the FP32-arithmetic reference; the native
probe compares against both before accepting any latency measurements.

The export and ONNX CPU comparison passed. The native NPU attempt uses the separate artifact directory
`npu-matte-grid-compact-flat-stable-broadcast-768-pipeline` and is still pending.
[Reformulation comparison](measurements/strix-halo-matte-production-equivalence.json),
[export configuration](measurements/strix-halo-matte-production-export.json),
[CPU and original FP16-alpha comparisons](measurements/strix-halo-matte-production-onnx-cpu.json).
Whole-stage NPU execution and performance remain unproven.

`python tools/rocm/bench_matte_stage.py` measures the actual preset's GPU stage from already-composited
CPU RGB through upload, normalization, resizing, FP16 inference and sigmoid, to working-size CPU alpha.
On example 2 its ten warm observations had median **140.263 ms** and p90 **140.519 ms**. Image loading,
compositing, SAM3, mask combination and model loading are excluded. The native NPU compiler was active
during collection, so this is a **provisional baseline**, not an uncontended GPU/NPU comparison; repeat
under matched load before claiming acceleration. [Raw observations](measurements/strix-halo-matte-production-gpu-provisional.json).
