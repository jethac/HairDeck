"""Capture the real decoder block whose fused offset transpose fails NPU tiling.

Includes offset and modulation convolutions so compiler fusion matches the full graph.
This diagnostic never changes the live GPU model.
"""
import argparse
import json
import sys
from functools import partial
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.prep.matte import build
from deform_grid import compact_deform_conv2d

parser = argparse.ArgumentParser()
parser.add_argument('--pack-height', action='store_true')
args = parser.parse_args()
name = 'decoder.decoder_block3.dec_att.aspp_deforms.1.atrous_conv'
root = Path(paths.results('npu-matte-atrous-768'+('-height' if args.pack_height else '')))
root.mkdir(parents=True, exist_ok=True)
model = build('birefnet_matting', 768, device='cuda').net.float()
block = dict(model.named_modules())[name]
source = Path(paths.results('npu-matte-grid-compact-flat-stable-broadcast-768-pipeline'))
x = torch.from_numpy(np.load(source/'reference.npz')['x']).cuda()
capture = {}

class Captured(Exception):
    pass

def grab(module, inputs):
    capture['features'] = inputs[0]
    raise Captured()

hook = block.register_forward_pre_hook(grab)
try:
    with torch.no_grad():
        model(x)
except Captured:
    pass
finally:
    hook.remove()
features = capture['features']
module = sys.modules[type(model).__module__]
original = module.deform_conv2d
with torch.no_grad():
    reference = block(features)
    module.deform_conv2d = partial(compact_deform_conv2d, stable_coordinates=True,
                                 pack_height=args.pack_height)
    try:
        actual = block(features)
        error = float((actual-reference).norm()/reference.norm())
        if not torch.isfinite(actual).all() or error > .001:
            raise RuntimeError(f'Decomposition changed block output: {error}')
        torch.onnx.export(block, (features,), str(root/'matte.onnx'),
                          input_names=['features'], output_names=['y'], opset_version=17, dynamo=False)
    finally:
        module.deform_conv2d = original
np.savez(root/'reference.npz', features=features.cpu().numpy(), y=reference.cpu().numpy())
metadata = dict(stage=name, source=str(source), input_shape=list(features.shape),
                pack_height=args.pack_height,
                output_shape=list(reference.shape), decomposition_relative_l2=error,
                reference_precision='FP32 arithmetic on production FP16-rounded weights')
(root/'export.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
