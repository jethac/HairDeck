"""Native Arch whole-stage feasibility probe; records partition profiling and rejects incorrect outputs.

Run with ryzen-ai-arch/scripts/run-sdk.sh. Host-array session.run timings include the
runtime transfer/interface costs, but not HairDeck preprocessing or cross-process scheduling.
"""
import argparse
import json
import time
import traceback
from pathlib import Path
import numpy as np
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--cpu-reference', action='store_true', help='Validate exported ONNX semantics without NPU compilation')
args = parser.parse_args()
root = Path(args.root)
config = {'passes': [{'name': 'init', 'plugin': 'vaip-pass_init'},
          {'name': 'vaiml_partition', 'plugin': 'vaip-pass_vaiml_partition',
           'vaiml_config': {'optimize_level': 1, 'preferred_data_storage': 'auto', 'tp_size': 1}}],
          'target': 'VAIML', 'targets': [{'name': 'VAIML', 'pass': ['init', 'vaiml_partition']}]}
(root/'bf16-single-stamp.json').write_text(json.dumps(config, indent=2))
report = dict(ort=ort.__version__, tp_size=1, cpu_reference=args.cpu_reference, success=False)
tag = 'cpu-reference' if args.cpu_reference else 'npu'
start = time.perf_counter()
try:
    opts = ort.SessionOptions()
    opts.log_severity_level = 1
    opts.intra_op_num_threads = 4
    opts.enable_profiling = True
    opts.profile_file_prefix = str(root/(tag+'-ort-profile'))
    provider_args = dict(providers=['VitisAIExecutionProvider'], provider_options=[{
            'config_file': str(root/'bf16-single-stamp.json'),
            'cache_dir': str(root/'cache-tp1'), 'enable_cache_file_io_in_mem': '0'}])
    if args.cpu_reference:
        provider_args = dict(providers=['CPUExecutionProvider'])
    session = ort.InferenceSession(str(root/'matte.onnx'), sess_options=opts, **provider_args)
    report['session_seconds'] = time.perf_counter()-start
    report['providers'] = session.get_providers()
    if not args.cpu_reference and 'VitisAIExecutionProvider' not in report['providers']:
        raise RuntimeError('Provider construction fell back to CPU')
    data = np.load(root/'reference.npz')
    feed = {session.get_inputs()[0].name: data['x']}
    y = session.run(None, feed)[-1]
    np.save(root/(tag+'-output.npy'), y)
    reference = data['y']
    sigmoid = lambda x: 1/(1+np.exp(-np.clip(x, -80, 80)))
    alpha, alpha_ref = sigmoid(y), sigmoid(reference)
    a, b = alpha > .5, alpha_ref > .5
    report['relative_l2'] = float(np.linalg.norm(y-reference)/np.linalg.norm(reference))
    report['alpha_mean_abs'] = float(np.abs(alpha-alpha_ref).mean())
    report['alpha_max_abs'] = float(np.abs(alpha-alpha_ref).max())
    report['mask_iou'] = float(np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b)))
    report['profile'] = session.end_profiling()
    events = json.loads(Path(report['profile']).read_text())
    placements = {}
    for event in events:
        provider = event.get('args', {}).get('provider')
        if provider:
            placements[provider] = placements.get(provider, 0)+1
    report['profile_provider_events'] = placements
    if not args.cpu_reference and not placements.get('VitisAIExecutionProvider'):
        raise RuntimeError('No VitisAI execution event; inspect compiler partition log before claiming NPU execution')
    if (not np.isfinite(y).all() or report['relative_l2'] > (.0001 if args.cpu_reference else .02)
            or report['alpha_mean_abs'] > .001 or report['mask_iou'] < .99):
        raise RuntimeError('Whole-stage numerical gate failed; no latency benchmark collected')
    if args.cpu_reference:
        report['success'] = True
        (root/(tag+'-result.json')).write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        raise SystemExit(0)
    for _ in range(3): session.run(None, feed)
    samples = []
    for _ in range(10):
        t = time.perf_counter()
        session.run(None, feed)
        samples.append((time.perf_counter()-t)*1000)
    report.update(success=True, samples_ms=samples, median_ms=float(np.median(samples)))
except Exception as error:
    report['error'] = f'{type(error).__name__}: {error}'
    (root/(tag+'-traceback.txt')).write_text(traceback.format_exc())
report['elapsed_seconds'] = time.perf_counter()-start
(root/(tag+'-result.json')).write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
raise SystemExit(0 if report['success'] else 2)
