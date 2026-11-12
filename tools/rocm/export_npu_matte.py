"""Whole BiRefNet ONNX export feasibility probe; no substituted operators or production changes.

Uses real 1024px model shape. Export success is not evidence of NPU operator coverage.
"""
import json
import sys
import time
import traceback
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.prep.matte import build

root = Path(paths.results('npu-matte-export'))
root.mkdir(parents=True, exist_ok=True)
model = build('birefnet_matting', 1024, device='cuda').net
x = torch.zeros(1, 3, 1024, 1024, device='cuda', dtype=torch.float16)
start = time.perf_counter()
result = dict(torch=torch.__version__, shape=list(x.shape), opset=17, exporter='legacy', success=False)
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
