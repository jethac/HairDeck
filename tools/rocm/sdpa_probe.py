"""Which SDPA backends work on this GPU, and how fast (bf16, SAM-3/DINOv3-like shapes)."""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def bench(fn, n=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

for B, H, L, D in [(1, 16, 5184, 64), (8, 8, 1024, 32), (1, 16, 1029, 64)]:
    q, k, v = (torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(q, k, v)
                ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
            print((B, H, L, D), be.name, '%.2f ms' % ms, 'maxerr %.3g' % (o.float() - ref).abs().max().item())
        except Exception as e:
            print((B, H, L, D), be.name, 'FAIL', type(e).__name__, str(e).splitlines()[0][:120])
