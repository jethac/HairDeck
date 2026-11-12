"""Measure actual preview rendering alongside generation on the shared Strix Halo GPU.

An isolated/overlapped pair, not a general memory-bandwidth benchmark. Uses fresh output
directories so the preview cache cannot turn the measured work into a file lookup.
"""
import concurrent.futures
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hairdeck import paths
from hairdeck.infer.image import Pipeline
from hairdeck.vrm.export import turn_preview

root = Path(tempfile.mkdtemp(prefix='overlap-', dir=paths.results()))
root.chmod(0o755)  # Keep bind-mounted results readable by the host user when Docker runs as root.
p = Pipeline()
image = paths.data('demo_examples', 'ex01.png')
warm = p.run(image, seed=1, out_dir=str(root/'warm'))
if not warm['ok']:
    raise RuntimeError('Warmup failed')

def generate(label):
    start = time.perf_counter()
    result = p.run(image, seed=1, n_samples=4, out_dir=str(root/label))
    if not result['ok']:
        raise RuntimeError('Generation failed')
    return dict(wall_seconds=time.perf_counter()-start, timings=result['timings'])

def preview(label):
    target = root/label
    shutil.copytree(root/'warm', target)
    start = time.perf_counter()
    result = turn_preview(str(target), 1, warm['hair_color'])
    return dict(wall_seconds=time.perf_counter()-start, result=result)

report = dict(root=str(root), isolated_generation=generate('isolated-generation'),
              isolated_preview=preview('isolated-preview'))
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    start = time.perf_counter()
    future = pool.submit(preview, 'overlapped-preview')
    report['overlapped_generation'] = generate('overlapped-generation')
    report['overlapped_preview'] = future.result()
    report['overlapped_pair_seconds'] = time.perf_counter()-start
(root/'timings.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
