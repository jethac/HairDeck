"""ROCm / gfx1151 smoke test: GPU torch, bf16 SDPA, EGL (pyrender) on radeonsi."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import torch
p = torch.cuda.get_device_properties(0)
print('torch', torch.__version__, 'hip', torch.version.hip, p.name, p.gcnArchName, 'total %.1f GiB' % (p.total_memory / 2**30))
print('mem_get_info GiB', [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()])
q = torch.randn(2, 16, 1024, 64, device='cuda', dtype=torch.bfloat16)
o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
print('sdpa ok', bool(torch.isfinite(o).all()))
import pyrender, trimesh
import numpy as np
r = pyrender.OffscreenRenderer(128, 128)
s = pyrender.Scene()
s.add(pyrender.Mesh.from_trimesh(trimesh.creation.box()))
cam = np.eye(4); cam[2, 3] = 3
s.add(pyrender.PerspectiveCamera(yfov=1.0), pose=cam)
s.add(pyrender.DirectionalLight(intensity=3), pose=cam)
c, d = r.render(s)
import OpenGL.GL as g
print('GL', g.glGetString(g.GL_RENDERER).decode(), '|', g.glGetString(g.GL_VERSION).decode(), '| depth px', int((d > 0).sum()))
r.delete()
