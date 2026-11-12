"""Capture a real BiRefNet deformable layer for focused NPU compatibility diagnosis.

This is a layer probe, not complete matting or a production model replacement.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.prep.matte import build
from deform_grid import compact_deform_conv2d

parser = argparse.ArgumentParser()
parser.add_argument('--stable-coordinates', action='store_true')
args = parser.parse_args()
root = Path(paths.results('npu-matte-deform-stable' if args.stable_coordinates else 'npu-matte-deform'))
root.mkdir(parents=True, exist_ok=True)
model = build('birefnet_matting', 1024, device='cuda').net.float()
x = torch.from_numpy(np.load(paths.results('npu-matte-grid', 'reference.npz'))['x']).cuda()
module = sys.modules[type(model).__module__]
original = module.deform_conv2d
capture = {}

class Captured(Exception):
    pass

def grab(**kwargs):
    capture.update(kwargs)
    raise Captured()

module.deform_conv2d = grab
try:
    with torch.no_grad(): model(x)
except Captured:
    pass
finally:
    module.deform_conv2d = original
if not capture:
    raise RuntimeError('No deformable layer was captured')

class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', capture['weight'].detach())
        self.register_buffer('bias', None if capture['bias'] is None else capture['bias'].detach())

    def forward(self, features, offset, mask):
        return compact_deform_conv2d(features, offset, self.weight, self.bias,
                                     stride=capture['stride'], padding=capture['padding'], mask=mask,
                                     stable_coordinates=args.stable_coordinates)

layer = Layer().eval()
inputs = tuple(capture[k] for k in ('input', 'offset', 'mask'))
with torch.no_grad():
    reference = original(**capture)
    actual = layer(*inputs)
    error = float((actual-reference).norm()/reference.norm())
    if not torch.isfinite(actual).all() or error > .001:
        raise RuntimeError(f'Layer decomposition changed output: {error}')
    torch.onnx.export(layer, inputs, str(root/'matte.onnx'), input_names=['features','offset','mask'],
                      output_names=['y'], opset_version=17, dynamo=False)
np.savez(root/'reference.npz', features=inputs[0].cpu().numpy(), offset=inputs[1].cpu().numpy(),
         mask=inputs[2].cpu().numpy(), y=reference.cpu().numpy())
metadata = dict(stage='first BiRefNet deformable convolution; real example-2 activations',
                stable_coordinates=args.stable_coordinates,
                shapes={k:list(capture[k].shape) for k in ('input','offset','weight','mask')},
                output_shape=list(reference.shape), decomposition_relative_l2=error)
(root/'export.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
