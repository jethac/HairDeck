"""Measure SAM3 stages and an optional compiled vision backbone on real examples."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from hairdeck import paths
from hairdeck.prep.preprocess import Preprocessor
from hairdeck.infer.image import load_image
p = Preprocessor('best')
model = p.prep.model
stages = {}
def timed(name, fn):
    def call(*a, **kw):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn(*a, **kw)
        torch.cuda.synchronize()
        stages[name] = stages.get(name, 0) + time.perf_counter() - start
        return result
    return call
model.backbone.forward_image = timed('vision', model.backbone.forward_image)
model.forward_grounding = timed('grounding', model.forward_grounding)
for name in ('_encode_prompt', '_run_encoder', '_run_decoder', '_run_segmentation_heads'):
    setattr(model, name, timed(name, getattr(model, name)))
images = [paths.data('demo_examples', f'ex{i:02}.png') for i in (1, 2, 3)]
def run(path, label):
    stages.clear()
    rgb, alpha, _, src = load_image(path, want_path=True)
    start = time.perf_counter()
    out = p(rgb, alpha, src)
    print(json.dumps(dict(label=label, image=Path(path).name, wall=time.perf_counter()-start, stages=stages.copy(), box=out['box'])), flush=True)
    return out
run(images[0], 'warmup')
base = [run(path, 'eager') for path in images]
if '--bf16-vision' in sys.argv:
    for module in model.backbone.vision_backbone.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(dtype=torch.bfloat16)
elif '--bf16-linears' in sys.argv:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(dtype=torch.bfloat16)
elif '--grounding' in sys.argv:
    for module in (model.transformer.encoder, model.transformer.decoder, model.segmentation_head):
        module.forward = torch.compile(module.forward, mode='default')
else:
    model.backbone.vision_backbone = torch.compile(model.backbone.vision_backbone, mode='default')
run(images[0], 'compile')
for path, original in zip(images, base):
    out = run(path, 'compiled')
    metrics = {}
    for key in ('hair', 'face', 'person'):
        a, b = original[key] > 127 if key == 'person' else original[key], out[key] > 127 if key == 'person' else out[key]
        metrics[key+'_iou'] = float((a & b).sum() / max(1, (a | b).sum()))
    print(json.dumps(dict(image=Path(path).name, metrics=metrics, same_box=original['box']==out['box'])), flush=True)
