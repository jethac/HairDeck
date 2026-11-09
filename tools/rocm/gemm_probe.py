"""bf16 GEMM throughput on this GPU for the set DiT's shapes (x @ W.T, as nn.Linear) and a large square reference.

python tools/rocm/gemm_probe.py
"""
import time

import torch
import torch.nn.functional as F


def tflops(M, N, K, n=50):
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
    for _ in range(3):
        F.linear(x, w)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n):
        F.linear(x, w)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n
    return 2 * M * N * K / dt / 1e12, dt * 1e6


print('M N K: TFLOPS, us')
for M in (384, 1536):
    for N, K in ((768, 768), (1536, 768), (3072, 768), (768, 3072), (6912, 768)):
        print(M, N, K, '%.1f TFLOPS %.0f us' % tflops(M, N, K))
for M in (2, 8):  # the adaLN projection of the timestep embedding: bandwidth-bound
    tf, us = tflops(M, 6912, 768)
    print(M, 6912, 768, '%.2f TFLOPS %.0f us  %.0f GB/s weights' % (tf, us, 6912 * 768 * 2 / us / 1e3))
print(4096, 4096, 4096, '%.1f TFLOPS %.0f us' % tflops(4096, 4096, 4096, 20))
