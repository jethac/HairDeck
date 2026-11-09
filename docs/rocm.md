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

Check the environment: `python tools/rocm/smoke.py` prints the GPU, a bf16 attention result and the EGL renderer
string, which should read `AMD Radeon 8060S Graphics (radeonsi, strix_halo, ACO, ...)`.

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

## Performance

PERF_PLACEHOLDER
