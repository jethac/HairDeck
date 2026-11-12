"""Standalone BiRefNet-matting latency on one demo image, fp16 at 1024 px.

The deployed 'best' preset uses 768 px; this historical standalone probe is not that stage's latency.

python tools/rocm/bench_matte.py [IMAGE]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from hairdeck import paths  # noqa: E402
from hairdeck.prep.matte import build  # noqa: E402

img = sys.argv[1] if len(sys.argv) > 1 else paths.data('demo_examples', 'ex01.png')
rgb = np.asarray(Image.open(img).convert('RGB'))
m = build('birefnet_matting', 1024, device='cuda')
for i in range(3):
    torch.cuda.synchronize(); t = time.perf_counter()
    a = m(rgb)
    torch.cuda.synchronize()
    print(f'run {i}: {(time.perf_counter() - t) * 1e3:.1f} ms  alpha mean {a.mean().item():.4f}')
if os.environ.get('PROFILE'):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        m(rgb); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))

if os.environ.get('COMPILE'):
    ref = m(rgb)
    m.net = torch.compile(m.net, dynamic=False)
    for i in range(4):
        torch.cuda.synchronize(); t = time.perf_counter()
        a = m(rgb)
        torch.cuda.synchronize()
        print(f'compiled run {i}: {(time.perf_counter() - t) * 1e3:.1f} ms  max|d alpha| {(a - ref).abs().max().item():.4f}')
