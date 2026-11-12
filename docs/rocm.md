# HairDeck on AMD ROCm (Strix Halo / gfx1151)

This fork runs the full HairDeck pipeline on AMD GPUs through ROCm. It was developed and validated on an AMD Ryzen AI
Max+ 395 ("Strix Halo") with the Radeon 8060S iGPU (gfx1151, RDNA 3.5, 40 CUs), 128 GiB LPDDR5X with 64 GiB assigned
to the GPU, on a Linux host with the `amdgpu` kernel driver.

ROCm builds of PyTorch expose the GPU as the `cuda` device, so HairDeck's `cuda` code paths run unchanged. The port
consists of the environment (below) and a faster sampler for bandwidth-bound GPUs (see Performance).

## Environment: Docker image

```bash
git clone -b rocm https://github.com/jethac/HairDeck.git && cd HairDeck
docker build -f docker/rocm-gfx1151.Dockerfile -t hairdeck-rocm .
mkdir -p ~/hairdeck-root ~/.cache/huggingface
docker run -d --name hairdeck --init --device=/dev/kfd --device=/dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --shm-size 16g -p 127.0.0.1:7860:7860 \
  -v "$PWD:/workspace/HairDeck" -v ~/hairdeck-root:/workspace/hairdeck-root \
  -v ~/.cache/huggingface:/root/.cache/huggingface hairdeck-rocm
docker exec -it hairdeck bash
```

Inside the container, follow the main README from "Model weights" on (`hf auth login`, then
`python scripts/setup/download_weights.py`). `HAIRDECK_ROOT` defaults to `/workspace/hairdeck-root`.

What the image contains:

| Component | Version | Why |
|---|---|---|
| Base | `vllm/vllm-openai-rocm` (digest pinned) | PyTorch 2.13 / torchvision 0.28 / Triton 3.8 built for ROCm 7.2.3 with gfx1151 kernels |
| Mesa | 26.2.5, built from source (`docker/build-mesa.sh`) | Ubuntu 22.04's Mesa 23.2 predates RDNA 3.5: `amdgpu: unknown (family_id, chip_external_rev): (150, 193)`, and `eglInitialize` fails. The build is headless radeonsi (EGL device / surfaceless, GBM, GL 4.6, ACO shaders, no LLVM), used by pyrender for the CHARM scan, colour targets and previews |
| libdrm | 2.4.134, built from source | Mesa 26.2 needs ≥ 2.4.133 |
| Python deps | `requirements-rocm.txt` (= `requirements.txt` without torch/torchvision) | installed into the image Python with `docker/rocm-constraints.txt` pinning the ROCm torch / torchvision / triton. In a venv, uv resolves a CUDA torch from PyPI instead |
| torchvision | 0.28.0 at the base image's commit, rebuilt with HIP kernels (`docker/build-torchvision.sh`) | the base image's torchvision has CPU-only ops: `torchvision::deform_conv2d` (BiRefNet-matting's decoder), `nms` and `roi_align` (SAM 3) fail on the GPU with `Could not run 'torchvision::...' with arguments from the 'CUDA' backend` |

### Persistent demo

Once weights and assets are installed, stop the setup container and use the Compose service. It runs the demo as
the container's foreground process and restarts after a crash or Docker/host restart. Docker itself must be enabled
at boot. It warms the pipeline before serving; inspect startup with `docker compose -f docker/compose.rocm.yml logs -f`.

```bash
export RENDER_GID=$(getent group render | cut -d: -f3)
export VIDEO_GID=$(getent group video | cut -d: -f3)
# Optional: HAIRDECK_IMAGE, HAIRDECK_DATA, HAIRDECK_HF_CACHE override the image and host directories.
docker compose -f docker/compose.rocm.yml up -d
```

`GRADIO_SERVER_NAME=0.0.0.0` binds inside the container; the published host port remains loopback-only. Use
`tailscale serve --bg --http=7860 http://127.0.0.1:7860` for tailnet access. The default standalone app binds localhost.

Check the environment: `python tools/rocm/smoke.py` prints the GPU, a bf16 attention result, the torchvision GPU ops
and the EGL renderer string, which should read `AMD Radeon 8060S Graphics (radeonsi, strix_halo, ACO, ...)`.

For another ROCm GPU, build with `--build-arg PYTORCH_ROCM_ARCH=gfx1100` (etc.); the base image's PyTorch must have
kernels for it. Mesa 26.2 supports every RDNA generation.

### What did not need porting

- **Attention.** PyTorch's flash and memory-efficient SDPA backends (AOTriton) have gfx1151 kernels, including the
  `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` that SAM 3's decoder forces (`tools/rocm/sdpa_probe.py`: ~20x the math
  backend, same error as it against an fp32 reference).
- **SAM 3's Triton kernels** (`sam3/model/edt.py`) are only used by its video tracker, not by HairDeck's image
  grounding. They are standard Triton and compile with AMD Triton anyway (ROCm tensors report `is_cuda`).
- **Model residency.** `torch.cuda.mem_get_info` reports the GPU carve-out (64 GiB here), so `HAIRDECK_OFFLOAD`
  stays off and every model stays resident.

## Validation

The released test-set evaluations run unchanged (`scripts/test.py`), seed 0, against the paper's three-seed means:

| Table 1, point clouds | CD | EMD | HD | IoU | Orient. |
|---|---|---|---|---|---|
| Paper (RTX A6000) | 0.0066 | 0.0084 | 0.0242 | 0.867 | 16.8° |
| ROCm gfx1151 | 0.0065 | 0.0084 | 0.0242 | 0.868 | 16.4° |

| Table 4, chains on artists' cards | CD | Rec. | Prec. | Motion | Ampl. | Fit |
|---|---|---|---|---|---|---|
| Paper (RTX A6000) | 0.279 | 0.625 | 0.585 | 0.702 | 0.0167 | 0.953 |
| ROCm gfx1151 | 0.277 | 0.620 | 0.596 | 0.701 | 0.0168 | 0.956 |
| ROCm gfx1151, fast sampler (below) | 0.277 | 0.620 | 0.598 | 0.702 | 0.0168 | 0.958 |

| Table 4, chains on the point-cloud model's cards | CD | Rec. | Prec. | Fit |
|---|---|---|---|---|
| Paper (RTX A6000) | 0.319 | 0.542 | 0.543 | 0.952 |
| ROCm gfx1151, fast sampler | 0.298 | 0.566 | 0.542 | 0.944 |

With the fast sampler Table 1 reads 0.0065 / 0.0084 / 0.0241 / 0.869 / 16.4°.

Other checks on gfx1151:
- BiRefNet-matting (fp16, GPU) against the same model in fp32 on the CPU (`tools/rocm/check_matte.py`): mean
  |Δalpha| 0.0002, mask IoU 0.999.
- VRM export and previews (`tools/rocm/make_vrm_fixture.py`, then `python -m hairdeck.vrm.export DIR 0`): Blender
  4.2 with the VRM Add-on, `hairdeck.vrm.check` passing, the rest-pose sheet and the head-turn video rendered through
  EGL on radeonsi.
- Not run here: the image pipeline end to end (SAM 3 and DINOv3 are gated downloads). Its parts run on the paths
  checked above: SAM 3 on flash SDPA and torchvision's HIP ops, the image DiT on the same set-DiT sampler as the
  point-cloud and chain DiTs.

## Performance

The set DiTs (point-cloud, image and chain models) sample 50 Euler steps with CFG 2. At batch 1 on Strix Halo their
GEMMs are small (M = 2 x 192 tokens) and a step is dominated by bandwidth-bound and launch-bound work rather than
matrix throughput. `hairdeck/sampling/fast.py` restructures the sampler, on by default on ROCm builds of PyTorch:

1. **Condition terms once per call.** The condition is fixed while sampling, so each block's cross-attention K/V
   (after their RMSNorm) and the pooled condition embedding are computed once instead of once per step.
2. **adaLN modulations for all steps in one GEMM per block.** `ada(silu(t_mlp(t) + pool))` depends only on the step;
   at M = 2 that projection reads 10.6 MB of weights per block and step at ~150 GB/s.
3. **bf16 weight copies.** Made once per model, instead of autocast casting every fp32 weight on every call.
4. **`torch.compile` of the step.** Inductor fuses the LayerNorms, RMSNorms, modulations, GELU, residuals and CFG
   combination between the GEMMs into Triton kernels for gfx1151. In the eager step these small kernels take about
   half the time.

Point-cloud model, batch 1, cards from the Dora condition (`tools/rocm/bench_pc.py`, warm):

| | sampling + decode | vs original |
|---|---|---|
| Original sampler (`HAIRDECK_FAST=0`) | 0.67 s | |
| 1-3, eager (`HAIRDECK_COMPILE=0`) | 0.45 s | 1.5x |
| 1-4 (default) | **0.31 s** | **2.2x** |

At batch 4: 1.85 s, 1.32 s and 0.87 s. The paper reports 1.01 s per hairstyle from the point cloud on an RTX A6000.

Numerics: with one step, the restructured forward is bitwise identical to `SetDiT.forward` under the same autocast
(`tools/rocm/check_fast.py`). Batching the adaLN GEMM over steps and fusing with inductor change the rounding, well
below autocast's own deviation from fp32 (0.6-1% of |v|). Over 50 steps that gives a different but equally likely
sample: the test-set metrics above are unchanged.

The first sampling call per model and batch size compiles (about a minute cold; a few seconds from the inductor cache in
`$HAIRDECK_ROOT/.cache/inductor`). The first BiRefNet call (MIOpen's convolution search, cached in
`$HAIRDECK_ROOT/.cache/miopen`) takes about 40 s once. The demo pays both once per container.

Settings: `HAIRDECK_FAST=0` restores the release sampler; `HAIRDECK_COMPILE=0` keeps 1-3 without compilation (and is
the fallback if compilation fails); `HAIRDECK_GRAPHS=1` also captures the step in a HIP graph.

Tried and not adopted:
- **HIP graph replay** of the step: 3% faster than plain launches without compilation and slower with it, on ROCm
  7.2. It also deadlocks when PyTorch TunableOp is enabled, because hipBLASLt cannot tune or look up inside a capture.
  Off by default; always off under TunableOp.
- **PyTorch TunableOp** (`tools/rocm/tune_gemm.sh`): rocBLAS / hipBLASLt solutions tuned for the 29 GEMM shapes moved
  the step by less than the run-to-run spread.
- **`TORCH_BLAS_PREFER_HIPBLASLT=0`**: rocBLAS is faster than hipBLASLt for some of these shapes in isolation
  (`tools/rocm/gemm_probe.py`), within noise end to end.
- **Forcing the GPU's DPM level to `high`**: no change; under load the 8060S already runs at 2.3-2.9 GHz.
- **Compiling BiRefNet**: 270 to 210 ms per image, for 56 s of compilation and up to 0.05 alpha difference (fp16
  reassociation) in a once-per-image step. Left eager.

## Repeated-image cache (2026-11-12)

ROCm pipelines retain one image's preprocessing result and one crop's DINO features. This avoids rerunning SAM3 and
BiRefNet when changing seeds and shares DINO features between colors and variations. The cache is bounded to one
entry per pipeline; image pixels (including alpha) determine reuse. Per-call custom preprocessing bypasses the mask
cache. Offloading and ZeroGPU disable caching. Set `HAIRDECK_IMAGE_CACHE=0` to disable it, or call
`Pipeline.clear_image_cache()` after changing weights or preprocessing settings on an existing pipeline.

Measured with `python tools/rocm/bench_image_cache.py`, example ex01, 50 steps, colors, chains and GLB export:
uncached warm request 1.850 s; first cached request 1.796 s; repeated image 0.685 s. Preprocessing fell from 1.066 s
to 0.002 s. Card tokens, colored-card JSON and chain JSON for the same seed were identical. These are single-run
measurements on the deployed Strix Halo, not a broad throughput benchmark. The generation subtotal now excludes color
time, which was previously also counted in that subtotal; total wall time was unaffected by that reporting error.

## Further optimization measurements (2026-11-12)

`tools/rocm/bench_sam3.py` measures real example preprocessing, vision, grounding and mask agreement. On examples 1-3,
eager preprocessing took 1.04-1.08 s. Compiling the vision backbone took 18.3 s on the first call and reduced warm
preprocessing to 1.00 s. Hair/face/person mask IoUs were 0.9973-0.9999 with unchanged crop boxes. Compiling grounding
components (`--grounding`) took 37.8 s initially and reduced preprocessing to 0.96-0.97 s; mask IoUs were
0.9994-0.99998 with unchanged boxes. Both remain experiments rather than default behavior: modest gains and nonzero
mask changes need a deliberate quality/performance choice. The probe retains stage timing for further investigation.

Preview video frames now stream directly to ffmpeg, avoiding PNG compression, temporary files and PNG decoding.
On the cache benchmark's 94-card, 24-chain example, the 90-frame preview took 2.41 s versus 3.58 s through PNG files;
render/stream time was 1.23 s and encoder flush 0.05 s. Encoding overlaps rendering, so the latter is not total encoding
CPU time. A diagnostic sent the same rendered frames through both paths and confirmed identical decoded RGB video
hashes. Separate rendering runs are not bitwise deterministic and are unsuitable for testing encoding equivalence.

The target also exposes its NPU (`amdxdna`, `/dev/accel/accel0`). AMD's Linux Ryzen AI documentation now describes
CNN INT8/BF16 and encoder NLP BF16 compilation: https://ryzenai.docs.amd.com/en/latest/linux.html . NPU use remains
unimplemented: operator coverage, model conversion, precision effects and end-to-end benefit must be established.
