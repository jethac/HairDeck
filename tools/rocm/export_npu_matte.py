"""Whole BiRefNet ONNX export feasibility probe; no production changes.

--decompose uses a fixed-shape GridSample adapter and checks real-image FP32 output
against the unchanged operator before export. Export success is not NPU coverage.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import torch
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.prep.matte import build

parser = argparse.ArgumentParser()
parser.add_argument('--decompose', action='store_true')
parser.add_argument('--compact-grid', action='store_true', help='Use one GridSample per deformable convolution')
args = parser.parse_args()
args.decompose = args.decompose or args.compact_grid
root = Path(paths.results('npu-matte-grid-compact' if args.compact_grid else
                         'npu-matte-grid' if args.decompose else 'npu-matte-export'))
root.mkdir(parents=True, exist_ok=True)
model = build('birefnet_matting', 1024, device='cuda').net
x = torch.zeros(1, 3, 1024, 1024, device='cuda', dtype=torch.float16)
if args.decompose:
    from deform_grid import deform_conv2d
    if args.compact_grid:
        from deform_grid import compact_deform_conv2d as deform_conv2d
    import torch.nn.functional as F
    model.float()
    rgb = np.asarray(Image.open(paths.data('demo_examples', 'ex02.png')).convert('RGB')).copy()
    x = torch.from_numpy(rgb).cuda().permute(2, 0, 1)[None].float()/255
    x = F.interpolate(x, (1024, 1024), mode='bilinear', align_corners=False)
    x = (x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.no_grad():
        reference = model(x)[-1].float()
        module = sys.modules[type(model).__module__]
        module.deform_conv2d = deform_conv2d
        decomposed = model(x)[-1].float()
    delta = decomposed-reference
    metrics = dict(relative_l2=float(delta.norm()/reference.norm()), max_abs=float(delta.abs().max()),
                   alpha_mean_abs=float((decomposed.sigmoid()-reference.sigmoid()).abs().mean()),
                   alpha_max_abs=float((decomposed.sigmoid()-reference.sigmoid()).abs().max()))
    (root/'equivalence.json').write_text(json.dumps(metrics, indent=2))
    print('EQUIVALENCE', json.dumps(metrics), flush=True)
    if not torch.isfinite(decomposed).all() or metrics['relative_l2'] > 0.001:
        raise RuntimeError('Decomposition changed real model output too much; do not export')
    np.savez(root/'reference.npz', x=x.cpu().numpy(), y=reference.cpu().numpy())
start = time.perf_counter()
result = dict(torch=torch.__version__, shape=list(x.shape), dtype=str(x.dtype),
              decomposition=args.decompose, compact_grid=args.compact_grid,
              opset=17, exporter='legacy', success=False)
try:
    with torch.no_grad():
        torch.onnx.export(model, (x,), str(root/'matte.onnx'), opset_version=17, dynamo=False)
    result['success'] = True
except Exception as error:
    result['error'] = f'{type(error).__name__}: {error}'
    (root/'traceback.txt').write_text(traceback.format_exc())
result['seconds'] = time.perf_counter()-start
(root/'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
