# Continued Strix Halo optimization after the NPU workaround

The NPU's corrected BF16 MLP is still slower than the compiled GPU MLP. Working arithmetic alone does not
justify inserting it into the GPU sampler. This pass measures batching and repeated GPU work, and probes
whether the existing independent matting stage can reach the NPU compiler.

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
can be inferred from this overlapping compilation experiment. Placement and NPU correctness remain pending.
