"""Measure the deployed preset's complete GPU matting stage, from CPU RGB to working-size CPU alpha.

Independent stage benchmark; excludes SAM3, recipe/mask combination, model loading and disk image loading.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.prep.matte import build
from hairdeck.prep.preprocess import BG, load_image

preset = json.loads((Path(__file__).resolve().parents[2]/'hairdeck/prep/presets.json').read_text())['best']
size, working = preset['alpha_size'], preset['wr']
_, image, _ = load_image(paths.data('demo_examples','ex02.png'), True, BG['black'])
rgb = np.asarray(image).copy()
h, w = rgb.shape[:2]
scale = min(1., working/max(h,w))
shape = (max(1,round(h*scale)), max(1,round(w*scale)))
model = build(preset['alpha'], size, device='cuda')

def run():
    alpha = model(rgb)
    return F.interpolate(alpha[None,None], shape, mode='bilinear', align_corners=False)[0,0].cpu().numpy()

for _ in range(3): result = run()
torch.cuda.synchronize()
samples = []
for _ in range(10):
    start = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    samples.append((time.perf_counter()-start)*1000)
report = dict(torch=torch.__version__, hip=torch.version.hip, preset='best', image='ex02.png',
              input_shape=list(rgb.shape), output_shape=list(result.shape), model_size=size,
              dtype='float16', boundary='CPU composited RGB -> upload/normalize/resize -> GPU model/sigmoid -> original-size GPU alpha -> working-size CPU alpha',
              excluded=['model loading','image loading/compositing','SAM3','mask recipe'],
              samples_ms=samples, median_ms=float(np.median(samples)), p90_ms=float(np.percentile(samples,90)))
root = Path(paths.results('matte-stage-gpu')); root.mkdir(exist_ok=True)
np.save(root/'alpha.npy',result)
(root/'result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
