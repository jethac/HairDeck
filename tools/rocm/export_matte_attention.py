"""Reduce the full-model compiler buffer error to its real second-scale Swin attention module."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.prep.matte import build

parser = argparse.ArgumentParser()
parser.add_argument('--flat-heads', action='store_true')
args = parser.parse_args()
root = Path(paths.results('npu-matte-attention-flat' if args.flat_heads else 'npu-matte-attention'))
root.mkdir(parents=True, exist_ok=True)
model = build('birefnet_matting', 1024, device='cuda').net.float()
attention = dict(model.named_modules())['bb.layers.1.blocks.0.attn']
x = torch.from_numpy(np.load(paths.results('npu-matte-grid', 'reference.npz'))['x']).cuda()
capture = {}
calls = 0

class Captured(Exception):
    pass

def grab(module, args, kwargs):
    global calls
    calls += 1
    if calls == 2:
        capture.update(features=args[0], mask=kwargs.get('mask', args[1] if len(args)>1 else None))
        raise Captured()

hook = attention.register_forward_pre_hook(grab, with_kwargs=True)
try:
    with torch.no_grad(): model(x)
except Captured:
    pass
finally:
    hook.remove()
if not capture:
    raise RuntimeError('Second-scale attention was not reached')

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = attention
        self.register_buffer('mask', capture['mask'])

    def forward(self, features):
        if args.flat_heads:
            from attention_layout import flat_window_attention
            return flat_window_attention(self.attention, features, self.mask)
        return self.attention(features, mask=self.mask)

layer = Attention().eval()
features = capture['features']
with torch.no_grad():
    reference = attention(features, mask=capture['mask'])
    actual = layer(features)
    error = float((actual-reference).norm()/reference.norm())
    if not torch.isfinite(actual).all() or error > .001:
        raise RuntimeError(f'Attention reformulation changed output: {error}')
    torch.onnx.export(layer, (features,), str(root/'matte.onnx'), input_names=['features'],
                      output_names=['y'], opset_version=17, dynamo=False)
np.savez(root/'reference.npz', features=features.cpu().numpy(), y=reference.cpu().numpy())
metadata = dict(stage='BiRefNet bb.layers.1.blocks.0.attn, second image scale, real example-2 activations',
                flat_heads=args.flat_heads, reformulation_relative_l2=error,
                input_shape=list(features.shape), output_shape=list(reference.shape),
                mask_shape=None if capture['mask'] is None else list(capture['mask'].shape))
(root/'export.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
