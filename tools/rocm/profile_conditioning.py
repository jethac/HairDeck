"""Focused real-device timing of repeated image conditioning; no sampler instrumentation."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import dev, paths
from hairdeck.infer.image import Pipeline
from hairdeck.sampling.fast import _Denoiser

p = Pipeline()
crop = np.asarray(Image.open(paths.results('variation-benchmark', '4-sequential', 'model_input.png')))
report = dict(torch=torch.__version__, hip=torch.version.hip, repetitions=20, timings={})
with torch.no_grad(), dev.amp(p.dev):
    features = p.image_features(crop)
    ctx = p.model.encode_cond(features)
    ts = torch.linspace(1, 0, 51, device=p.dev)[:-1, None].contiguous()
    for name, fn in [('cached_image_features', lambda: p.image_features(crop)),
                     ('encode_cond', lambda: p.model.encode_cond(features)),
                     ('denoiser_prepare', lambda: _Denoiser(p.model, ctx, 2.0, ts))]:
        for _ in range(3): result = fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        report['timings'][name] = dict(median_ms=float(np.median(times)), samples_ms=times)
print(json.dumps(report, indent=2))
