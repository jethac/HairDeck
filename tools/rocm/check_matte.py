"""BiRefNet-matting on the GPU (fp16, as the release) against the same model in fp32 on the CPU, on one demo image.

python tools/rocm/check_matte.py [IMAGE]
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
t = time.perf_counter()
g = build('birefnet_matting', 1024, device='cuda')
a_gpu = g(rgb).float().cpu()
print(f'gpu first call (incl. load) {time.perf_counter() - t:.1f} s')
c = build('birefnet_matting', 1024, device='cpu')
c.net.float()
with torch.no_grad():
    t0 = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
    t0 = torch.nn.functional.interpolate(t0, (1024, 1024), mode='bilinear', align_corners=False)
    t0 = (t0 - c.mean) / c.std
    a = c.net(t0)[-1].sigmoid()
    a_cpu = torch.nn.functional.interpolate(a, rgb.shape[:2], mode='bilinear', align_corners=False)[0, 0]
d = (a_gpu - a_cpu).abs()
print(f'alpha gpu fp16 vs cpu fp32: mean |d| {d.mean():.4f}  p99 {d.flatten().kthvalue(int(0.99 * d.numel())).values:.4f}  '
      f'mask IoU(>0.5) {((a_gpu > .5) & (a_cpu > .5)).sum() / ((a_gpu > .5) | (a_cpu > .5)).sum():.4f}')
