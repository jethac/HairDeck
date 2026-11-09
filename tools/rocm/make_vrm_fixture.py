"""A saved-result directory as hairdeck.infer.image writes one (cards_seed0.json, chains_seed0.json in the head frame,
info.json), made from the test-set outputs instead of an image: the point-cloud model's cards of test hairstyle H
(scripts/test.py pointcloud) and the chains generated on them (scripts/test.py chains --cards <that run>). For checking
the VRM export (Blender + VRM Add-on) and the head-turn preview without the gated SAM 3 / DINOv3 weights.

python tools/rocm/make_vrm_fixture.py PC_RUN CHAINS_RUN H OUT_DIR
then: python -m hairdeck.vrm.export OUT_DIR 0
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from hairdeck import paths  # noqa: E402

pc_run, ch_run, h, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
hz = np.load(paths.data('head_frame_test.npz'))
k, b = float(hz['k'][h]), hz['b'][h].astype(np.float64)
cards = json.load(open(f'{paths.resolve(pc_run)}/JsonResults/{h:03d}.json'))
for c in cards:  # dataset frame -> head frame (the inverse of hairdeck.vrm.prep.head_to_body)
    q = np.asarray(c['seq'], np.float64)
    q[:, :3] = q[:, :3] * k + b
    q[:, 3:5] *= k
    c['seq'] = q.tolist()
chains = json.load(open(f'{paths.resolve(ch_run)}/chains.json'))[str(h)]['pred']
for c in chains:
    for key in ('joints', 'path'):
        if key in c:
            c[key] = (np.asarray(c[key], np.float64) * k + b).tolist()
os.makedirs(out, exist_ok=True)
json.dump(cards, open(f'{out}/cards_seed0.json', 'w'))
json.dump(chains, open(f'{out}/chains_seed0.json', 'w'))
json.dump(dict(hair_color=[0.32, 0.22, 0.16], samples=[dict(seed=0, n_meshed=len(cards), n_chains=len(chains))]),
          open(f'{out}/info.json', 'w'), indent=1)
print(f'{out}: {len(cards)} cards, {len(chains)} chains (test hairstyle {h})')
