"""Fast Euler/CFG sampling of the set DiTs (hairdeck.models.setdit.SetDiT and its subclasses), the same operations as
hairdeck.sampling.flow.fm_sample / hairdeck.infer.image.euler_cfg with Euler steps, restructured for small batches:

- the condition is fixed during sampling, so each block's cross-attention keys / values (after their RMSNorm) and the
  pooled condition embedding are computed once per call instead of once per step;
- the Linear weights are read from a bf16 copy made once per model, the cast bf16 autocast makes of every weight in
  every call (an fp32 read and a bf16 write of each weight per step, on hardware where the denoiser is bound by memory
  bandwidth, such as the LPDDR5X of AMD Strix Halo);
- one Euler step (the doubled CFG batch, the guidance combination and the update) is captured in a CUDA / HIP graph and
  replayed, so the ~700 small kernels of a step are launched without per-kernel CPU overhead, and the per-step host
  sync of float(t) goes away.
Every tensor operation is the one the module would run under the same autocast, in the same batch shapes, so the
result is the original's (bitwise in our checks on ROCm gfx1151; tools/rocm/check_fast.py).

HAIRDECK_FAST: 1 on, 0 off; default on for ROCm builds of PyTorch, off otherwise (the CUDA release keeps its own
path). HAIRDECK_GRAPHS=0 keeps the fast path without graph capture.
"""
import os

import torch
import torch.nn.functional as F

from ..models.setdit import SetDiT, modulate, timestep_embedding


def enabled(model):
    want = os.environ.get('HAIRDECK_FAST')
    on = bool(torch.version.hip) if want is None else want == '1'
    # Only the base forward is restructured here; a subclass with its own forward keeps the generic path.
    return (on and isinstance(model, SetDiT) and type(model).forward is SetDiT.forward
            and next(model.parameters()).device.type == 'cuda' and torch.is_autocast_enabled('cuda')
            and torch.get_autocast_dtype('cuda') == torch.bfloat16)


def _graphs_on():
    return os.environ.get('HAIRDECK_GRAPHS', '1') == '1'


class _Weights:
    """bf16 copies of a SetDiT's block Linear weights (autocast would cast them in every call), kept with the module
    and refreshed if its parameters change (training, load_state_dict, .to())."""

    def __init__(self, model):
        self.key = self._key(model)
        self.w = {}
        for b in model.blocks:
            for m in b.modules():
                if isinstance(m, torch.nn.Linear):
                    self.w[m] = (m.weight.detach().to(torch.bfloat16),
                                 None if m.bias is None else m.bias.detach().to(torch.bfloat16))
        for m in (model.inp, *model.t_mlp, *model.pool_mlp):
            if isinstance(m, torch.nn.Linear):
                self.w[m] = (m.weight.detach().to(torch.bfloat16),
                             None if m.bias is None else m.bias.detach().to(torch.bfloat16))

    @staticmethod
    def _key(model):
        p = model.blocks[0].ada.weight
        return p.data_ptr(), p._version, p.device, sum(q._version for q in model.parameters())

    def fresh(self, model):
        return self._key(model) == self.key


def _weights(model):
    w = getattr(model, '_hairdeck_fast_w', None)
    if w is None or not w.fresh(model):
        w = _Weights(model)
        object.__setattr__(model, '_hairdeck_fast_w', w)  # not a submodule / buffer: stays out of state_dict
    return w


class _Denoiser:
    def __init__(self, model, ctx, cfg):
        self.m = model
        self.W = _weights(model).w
        B = ctx.shape[0]
        self.cfg = cfg
        if cfg != 1.0:  # the doubled batch of fm_sample: rows B.. use the learned null context
            ctx = torch.cat([ctx, ctx])
            drop = torch.cat([torch.zeros(B, dtype=torch.bool, device=ctx.device),
                              torch.ones(B, dtype=torch.bool, device=ctx.device)])
            ctx = torch.where(drop[:, None, None], model.null_ctx[None].to(ctx.dtype).expand_as(ctx), ctx)
        self.ctx = ctx
        self.pooled = self._lin_seq(model.pool_mlp, ctx.mean(1))
        self.kv = []
        for b in model.blocks:
            if b.cross:
                a = b.xattn
                M, C = ctx.shape[1], model.dim
                k, v = self.lin(a.kv, ctx).view(ctx.shape[0], M, 2, a.h, C // a.h).permute(2, 0, 3, 1, 4)
                self.kv.append((a.kn(k), v))
            else:
                self.kv.append(None)

    def lin(self, m, x):
        w, b = self.W[m]
        return F.linear(x, w, b)

    def _lin_seq(self, seq, x):
        for m in seq:
            x = self.lin(m, x) if isinstance(m, torch.nn.Linear) else m(x)
        return x

    def _attn(self, a, x, kv=None):
        B, N, C = x.shape
        q = self.lin(a.q, x).view(B, N, a.h, C // a.h).transpose(1, 2)
        if kv is None:
            k, v = self.lin(a.kv, x).view(B, N, 2, a.h, C // a.h).permute(2, 0, 3, 1, 4)
            k = a.kn(k)
        else:
            k, v = kv
        q = a.qn(q)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.lin(a.o, o.transpose(1, 2).reshape(B, N, C))

    def _mlp(self, mlp, x):
        return self._lin_seq(mlp, x)

    def _block(self, b, x, c, kv):
        m = self.lin(b.ada, F.silu(c)).chunk(9 if b.cross else 6, -1)
        cast = (lambda t: t.float()) if b.fp32_res else (lambda t: t)
        x = cast(x) + cast(m[2][:, None] * self._attn(b.attn, modulate(b.n1(x), m[0], m[1])))
        if b.cross:
            x = x + cast(m[5][:, None] * self._attn(b.xattn, modulate(b.n2(x), m[3], m[4]), kv))
            x = x + cast(m[8][:, None] * self._mlp(b.mlp, modulate(b.n3(x), m[6], m[7])))
        else:
            x = x + cast(m[5][:, None] * self._mlp(b.mlp, modulate(b.n3(x), m[3], m[4])))
        return x

    def forward(self, x, t):
        """SetDiT.forward(x, t, ctx, drop) for the (doubled) batch, with the cached condition terms."""
        m = self.m
        c = self._lin_seq(m.t_mlp, timestep_embedding(t, 256)) + self.pooled
        h = self.lin(m.inp, x).float() if m.fp32_res else self.lin(m.inp, x)
        for b, kv in zip(m.blocks, self.kv):
            h = self._block(b, h, c, kv)
        with torch.autocast('cuda', enabled=False):
            c, h = c.float(), h.float()
            sh, sc = m.f_ada(F.silu(c)).chunk(2, -1)
            return m.out(modulate(m.fn(h), sh, sc))

    def step(self, x, t, dt):
        """One Euler step of fm_sample: x + (tn - t) * v with v the CFG combination (float32)."""
        B = x.shape[0]
        if self.cfg == 1.0:
            v = self.forward(x, t).float()
        else:
            vc, vu = self.forward(torch.cat([x, x]), torch.cat([t, t])).float().chunk(2)
            v = vu + self.cfg * (vc - vu)
        return x + dt * v


def euler(model, ctx, x, steps, cfg):
    """Euler integration from the noise x (t = 1) to t = 0 in `steps` uniform steps with CFG `cfg`, as fm_sample."""
    B = x.shape[0]
    d = _Denoiser(model, ctx, cfg)
    ts = torch.linspace(1, 0, steps + 1, device=ctx.device)
    # fm_sample's per-step scalars: t as torch.full((B,), float(t)), the update (tn - t) in float32
    T = ts[:-1, None].expand(steps, B).contiguous()
    DT = (ts[1:] - ts[:-1]).contiguous()
    if not _graphs_on():
        for i in range(steps):
            x = d.step(x, T[i], DT[i])
        return x
    xs, tb, dtb = x.clone(), T[0].clone(), DT[0].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm-up outside capture (lazy init, kernel selection)
        d.step(xs, tb, dtb)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = d.step(xs, tb, dtb)
    for i in range(steps):
        tb.copy_(T[i])
        dtb.copy_(DT[i])
        g.replay()
        xs.copy_(out)
    return xs.clone()
