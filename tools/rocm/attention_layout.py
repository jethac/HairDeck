"""Experimental equivalent Swin attention layout for fixed-shape ONNX export.

Adapted from the Microsoft Swin WindowAttention used by BiRefNet.
Copyright (c) 2021 Microsoft. Original code licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


def flat_window_attention(module, x, mask=None):
    """Fold window and head into one batched-matmul axis; retain every window/head."""
    if module.training:
        raise ValueError('Export adapter requires evaluation mode')
    b, n, c = map(int, x.shape)
    heads = module.num_heads
    qkv = module.qkv(x).reshape(b, n, 3, heads, c//heads).permute(2, 0, 3, 1, 4)
    qkv = qkv.reshape(3, b*heads, n, c//heads)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scores = (q*module.scale) @ k.transpose(-2, -1)
    bias = module.relative_position_bias_table[module.relative_position_index.reshape(-1)]
    bias = bias.reshape(n, n, heads).permute(2, 0, 1)
    # Broadcast in the original window/head axes. Explicitly repeating the bias
    # before flattening makes VAIML constant-fold gigabytes of duplicate values.
    scores = scores.reshape(b, heads, n, n)+bias[None]
    if mask is not None:
        nw = int(mask.shape[0])
        scores = scores.reshape(b//nw, nw, heads, n, n)+mask[None, :, None]
    scores = scores.reshape(b*heads, n, n)
    probs = module.attn_drop(module.softmax(scores))
    out = (probs @ v).reshape(b, heads, n, c//heads).transpose(1, 2).reshape(b, n, c)
    return module.proj_drop(module.proj(out))
