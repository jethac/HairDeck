"""Experimental fixed-shape BiRefNet export adapter; never installed in the production model.

BiRefNet uses one offset group and one convolution group. Preserve bilinear sampling,
zero padding, modulation and learned kernel weights with GridSample + 1x1 Conv.
"""
import torch
import torch.nn.functional as F


def deform_conv2d(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0),
                  dilation=(1, 1), mask=None):
    pair = lambda x: (x, x) if isinstance(x, int) else x
    stride, padding, dilation = map(pair, (stride, padding, dilation))
    # This export targets the one fixed 1024px pipeline shape. Concrete sizes also
    # let the legacy exporter infer the reshaped 1x1 convolution kernel.
    b, c, h, w = map(int, input.shape)
    out_c, in_c, kh, kw = map(int, weight.shape)
    oh, ow = map(int, offset.shape[-2:])
    assert in_c == c and offset.shape[1] == 2*kh*kw, 'Only BiRefNet single-group shapes supported'
    yy = torch.arange(oh, device=input.device, dtype=input.dtype).view(1, oh, 1)*stride[0]-padding[0]
    xx = torch.arange(ow, device=input.device, dtype=input.dtype).view(1, 1, ow)*stride[1]-padding[1]
    samples = []
    for ky in range(kh):
        for kx in range(kw):
            k = ky*kw+kx
            y = yy+ky*dilation[0]+offset[:, 2*k]
            x = xx+kx*dilation[1]+offset[:, 2*k+1]
            grid = torch.stack((2*(x+0.5)/w-1, 2*(y+0.5)/h-1), dim=-1)
            sample = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            if mask is not None:
                sample = sample*mask[:, k:k+1]
            samples.append(sample)
    sampled = torch.stack(samples, dim=2).reshape(b, c*kh*kw, oh, ow)
    return F.conv2d(sampled, weight.reshape(out_c, c*kh*kw, 1, 1), bias)
