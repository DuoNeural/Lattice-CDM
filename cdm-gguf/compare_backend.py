#!/usr/bin/env python3
"""Compare real CDM libllama logits against FP32 PyTorch, with optional KV-cache test."""
import argparse,json,os,pathlib,subprocess,tempfile,time
import numpy as np
import torch
from cdm_tools import load,forward,PRESETS,ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('gguf');p.add_argument('--runner',default=str(ROOT/'cdm-logits'));p.add_argument('--preset',choices=PRESETS);p.add_argument('--config');p.add_argument('--ids',default='1,2,3,4');p.add_argument('--threads',type=int,default=4);p.add_argument('--incremental',action='store_true');p.add_argument('--output');p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--rtol',type=float,default=1e-4);a=p.parse_args()
 config=dict(PRESETS.get(a.preset,{}))
 if a.config:config.update(json.loads(pathlib.Path(a.config).read_text()))
 torch.set_num_threads(a.threads);s,cfg=load(a.checkpoint,config);ids=torch.tensor([int(v) for v in a.ids.split(',')])
 with torch.inference_mode():expected=forward(s,cfg,ids).numpy()
 env=dict(os.environ,CDM_THREADS=str(a.threads));env['CDM_INCREMENTAL']='1' if a.incremental else '0'
 with tempfile.TemporaryDirectory() as d:
  out=pathlib.Path(d)/'logits.f32';r=subprocess.run([a.runner,a.gguf,str(out),*[str(x) for x in ids.tolist()]],env=env,capture_output=True,text=True,timeout=300)
  if r.returncode:raise RuntimeError(r.stderr)
  actual=np.fromfile(out,dtype=np.float32).reshape(expected.shape)
 delta=np.abs(actual-expected);ok=bool(np.allclose(actual,expected,atol=a.atol,rtol=a.rtol))
 result=dict(gguf=a.gguf,ids=ids.tolist(),incremental=a.incremental,threads=a.threads,shape=list(expected.shape),finite=bool(np.isfinite(actual).all()),max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()),rms_error=float(np.sqrt(np.mean(delta**2))),top1_agreement=float(np.mean(actual.argmax(-1)==expected.argmax(-1))),expected_top1=expected.argmax(-1).tolist(),actual_top1=actual.argmax(-1).tolist(),allclose=ok,atol=a.atol,rtol=a.rtol,backend_stderr=r.stderr)
 print(json.dumps(result,indent=2))
 if a.output:pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 if not ok:raise SystemExit(1)
if __name__=='__main__':main()
