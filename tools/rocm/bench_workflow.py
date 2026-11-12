"""Real deployed workflow timings. Preserves disk compiler caches; no unit tests.

Run in a fresh Python process. The first request includes process-local initialization,
not an empty-cache installation. Outputs and machine-readable timings stay under --root.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

started = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from hairdeck import paths
from hairdeck.infer.image import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--root', default=paths.results('workflow-benchmark'))
args = parser.parse_args()
root = Path(args.root)
root.mkdir(parents=True, exist_ok=True)
pipeline = Pipeline()
report = dict(python=platform.python_version(), torch=torch.__version__, rocm=torch.version.hip,
              device=torch.cuda.get_device_name(), startup_seconds=time.perf_counter()-started,
              pipeline_load_seconds=pipeline.load_sec, disk_caches='retained', rows=[])
print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
for label, image, count in [('first_request', 1, 1), ('repeat_image', 1, 1),
                            ('new_image', 2, 1), ('repeat_second_image', 2, 1),
                            ('four_variations', 2, 4), ('new_third_image', 3, 1)]:
    result = pipeline.run(paths.data('demo_examples', f'ex{image:02}.png'), seed=1, steps=50,
                          n_samples=count, colors=True, chains=True, out_dir=str(root/label))
    row = dict(label=label, image=image, variations=count, ok=result['ok'], timings=result['timings'],
               cache=result['cache'], notes=result['notes'],
               samples=[{k: s[k] for k in ('seed', 'n_cards', 'n_chains', 'colors')} for s in result['samples']],
               gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
               gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    report['rows'].append(row)
    (root/'timings.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(row), flush=True)
    if not result['ok'] or any(not s['colors'] or not s['n_chains'] for s in row['samples']):
        raise RuntimeError(f'{label}: workflow did not produce colors and chains; inspect saved output')
