"""Experimental real HairDeck MLP weights, representative synthetic activations; not full model inference.
export: CPU PyTorch + onnx + safetensors. gpu: ROCm PyTorch. npu: native Ryzen AI environment.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('mode', choices=['export','gpu','npu'])
p.add_argument('--root', required=True)
p.add_argument('--weights')
p.add_argument('--flat', action='store_true')
p.add_argument('--compile', action='store_true')
p.add_argument('--linear', action='store_true')
p.add_argument('--width',type=int,default=0)
p.add_argument('--layout',choices=['auto','unvectorized','vectorized'],default='unvectorized')
a=p.parse_args()
root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
if a.mode in ('export','gpu'):
    import torch
    from safetensors.torch import load_file
    sd=load_file(a.weights)
    w=sd['blocks.0.mlp.0.weight']; dim=w.shape[1]; hidden=w.shape[0]
    model=torch.nn.Sequential(torch.nn.Linear(dim,hidden),torch.nn.GELU(approximate='tanh'),torch.nn.Linear(hidden,dim)).eval()
    model.load_state_dict({k.removeprefix('blocks.0.mlp.'):v for k,v in sd.items() if k.startswith('blocks.0.mlp.')})
    del sd
    if a.linear:
        model=model[0]
        if a.width:
            model.weight=torch.nn.Parameter(model.weight[:a.width].clone())
            model.bias=torch.nn.Parameter(model.bias[:a.width].clone())
            model.out_features=a.width
    if a.mode=='export':
        torch.manual_seed(42)
        x=torch.randn(2,192,dim)
        if a.flat: x=x.reshape(-1,dim)
        with torch.no_grad(): y=model(x)
        np.savez(root/'reference.npz',x=x.numpy(),y=y.numpy())
        torch.onnx.export(model,(x,),str(root/'mlp.onnx'),input_names=['x'],output_names=['y'],opset_version=17,dynamo=False)
        print(json.dumps(dict(shape=list(x.shape),dim=dim,hidden=hidden,weights='blocks.0.mlp',input='synthetic standard normal')))
        raise SystemExit
    data=np.load(root/'reference.npz'); x=torch.from_numpy(data['x']).cuda(); model=model.cuda()
    if a.compile: model=torch.compile(model)
    def run():
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): return model(x)
    sync=torch.cuda.synchronize
else:
    import onnxruntime as ort
    config={'passes':[{'name':'init','plugin':'vaip-pass_init'}, {'name':'vaiml_partition','plugin':'vaip-pass_vaiml_partition','vaiml_config':{'optimize_level':1,'preferred_data_storage':a.layout}}], 'target':'VAIML','targets':[{'name':'VAIML','pass':['init','vaiml_partition']}]}
    (root/'bf16.json').write_text(json.dumps(config))
    opts=ort.SessionOptions(); opts.log_severity_level=1
    start=time.perf_counter()
    session=ort.InferenceSession(str(root/'mlp.onnx'),sess_options=opts,providers=['VitisAIExecutionProvider'],provider_options=[{'config_file':str(root/'bf16.json'),'cache_dir':str(root/('cache-'+a.layout)),'enable_cache_file_io_in_mem':'0'}])
    print('session_seconds',time.perf_counter()-start,flush=True)
    assert 'VitisAIExecutionProvider' in session.get_providers(), 'CPU-only fallback'
    data=np.load(root/'reference.npz')
    def run(): return session.run(None,{'x':data['x']})[0]
    def sync(): pass
for _ in range(5): result=run()
sync(); times=[]
for _ in range(50):
    t=time.perf_counter(); result=run(); sync(); times.append((time.perf_counter()-t)*1000)
if a.mode=='gpu': result=result.float().cpu().numpy()
np.save(root/(a.mode+'-output.npy'),result)
ref=data['y']; metrics=dict(mode=a.mode,compiled=a.compile,median_ms=float(np.median(times)),p90_ms=float(np.percentile(times,90)),relative_l2=float(np.linalg.norm(result-ref)/np.linalg.norm(ref)),max_absolute_error=float(np.max(np.abs(result-ref))),shape=list(ref.shape))
print(json.dumps(metrics),flush=True)
(root/(a.mode+('-compiled' if a.compile else '')+'-metrics.json')).write_text(json.dumps(metrics,indent=2))
