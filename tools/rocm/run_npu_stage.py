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
parser.add_argument('--validate-only', action='store_true', help='Check placement and output, defer timing during other experiments')
parser.add_argument('--feature-output', action='store_true', help='Layer diagnostic: apply L2 gate, not whole-matte alpha/IoU gates')
parser.add_argument('--layout', choices=['auto','vectorized','unvectorized'], default='auto')
args = parser.parse_args()
root = Path(args.root)
config = {'passes': [{'name': 'init', 'plugin': 'vaip-pass_init'},
          {'name': 'vaiml_partition', 'plugin': 'vaip-pass_vaiml_partition',
           'vaiml_config': {'optimize_level': 1, 'preferred_data_storage': args.layout, 'tp_size': 1}}],
          'target': 'VAIML', 'targets': [{'name': 'VAIML', 'pass': ['init', 'vaiml_partition']}]}
config_path = root/f'bf16-single-stamp-{args.layout}.json'
config_path.write_text(json.dumps(config, indent=2))
report = dict(ort=ort.__version__, tp_size=1, cpu_reference=args.cpu_reference,
              validate_only=args.validate_only, feature_output=args.feature_output, layout=args.layout, success=False)
tag = 'cpu-reference' if args.cpu_reference else 'npu'
if args.layout != 'auto': tag += '-'+args.layout
start = time.perf_counter()
try:
    opts = ort.SessionOptions()
    opts.log_severity_level = 1
    opts.intra_op_num_threads = 4
    opts.enable_profiling = True
    opts.profile_file_prefix = str(root/(tag+'-ort-profile'))
    provider_args = dict(providers=['VitisAIExecutionProvider'], provider_options=[{
            'config_file': str(config_path),
            'cache_dir': str(root/('cache-tp1' if args.layout == 'auto' else 'cache-tp1-'+args.layout)),
            'enable_cache_file_io_in_mem': '0'}])
    if args.cpu_reference:
        provider_args = dict(providers=['CPUExecutionProvider'])
    session = ort.InferenceSession(str(root/'matte.onnx'), sess_options=opts, **provider_args)
    report['session_seconds'] = time.perf_counter()-start
    report['providers'] = session.get_providers()
    if not args.cpu_reference and 'VitisAIExecutionProvider' not in report['providers']:
        raise RuntimeError('Provider construction fell back to CPU')
    data = np.load(root/'reference.npz')
    feed = {entry.name: data[entry.name] if entry.name in data else data['x']
            for entry in session.get_inputs()}
    y = session.run(None, feed)[-1]
    np.save(root/(tag+'-output.npy'), y)
    reference = data['y']
    report['relative_l2'] = float(np.linalg.norm(y-reference)/max(1e-12, float(np.linalg.norm(reference))))
    alpha_ok = True
    if not args.feature_output:
        sigmoid = lambda x: 1/(1+np.exp(-np.clip(x, -80, 80)))
        alpha, alpha_ref = sigmoid(y), sigmoid(reference)
        a, b = alpha > .5, alpha_ref > .5
        report['alpha_mean_abs'] = float(np.abs(alpha-alpha_ref).mean())
        report['alpha_max_abs'] = float(np.abs(alpha-alpha_ref).max())
        report['mask_iou'] = float(np.count_nonzero(a & b)/max(1, np.count_nonzero(a | b)))
        alpha_ok = report['alpha_mean_abs'] <= .001 and report['mask_iou'] >= .99
        if 'production_alpha' in data:
            target = data['production_alpha']
            target_mask = target > .5
            report['production_alpha_mean_abs'] = float(np.abs(alpha-target).mean())
            report['production_alpha_max_abs'] = float(np.abs(alpha-target).max())
            report['production_mask_iou'] = float(np.count_nonzero(a & target_mask)/max(1, np.count_nonzero(a | target_mask)))
            alpha_ok = alpha_ok and report['production_alpha_mean_abs'] <= .001 and report['production_mask_iou'] >= .99
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
    if (not np.isfinite(y).all() or not np.isfinite(report['relative_l2'])
            or report['relative_l2'] > (.0001 if args.cpu_reference else .02)
            or not alpha_ok):
        raise RuntimeError('Numerical gate failed; no latency benchmark collected')
    if args.cpu_reference or args.validate_only:
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
