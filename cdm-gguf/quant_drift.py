#!/usr/bin/env python3
"""Evaluate Q8 logit/distribution drift on multiple exact token prefixes.
Acceptance is deliberately distributional: max absolute logits <=0.5, mean KL
<=0.01 nats, and aggregate token top-1 agreement >=90%. This is a smoke test,
not a language quality benchmark; teacher-tokenizer text eval is still required.
"""
import argparse,json,os,pathlib,subprocess,tempfile
import numpy as np
import torch
from cdm_tools import ROOT,PRESETS,load,forward

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runner',default=str(ROOT/'cdm-logits'));p.add_argument('checkpoint');p.add_argument('gguf');p.add_argument('--preset',choices=PRESETS,required=True);p.add_argument('--output');p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);s,cfg=load(a.checkpoint,PRESETS[a.preset])
 cases=[[1,2,3,4],[198,520,198,520,42,17,198,284],[17,28,42,327,355,281,17,28,42,327,355,281,17,28,42,327]];records=[];agreements=[];kls=[];maximum=0
 for case in cases:
  ids=torch.tensor(case)
  with torch.inference_mode():ref=forward(s,cfg,ids)
  with tempfile.TemporaryDirectory() as d:
   dest=pathlib.Path(d)/'out.f32';env=dict(os.environ,CDM_THREADS=str(a.threads),CDM_INCREMENTAL='1');r=subprocess.run([a.runner,a.gguf,str(dest),*map(str,case)],env=env,capture_output=True,text=True,check=True,timeout=300);actual=torch.from_numpy(np.fromfile(dest,dtype=np.float32).reshape(tuple(ref.shape)))
  delta=(actual-ref).abs();kl=(ref.softmax(-1)*(ref.log_softmax(-1)-actual.log_softmax(-1))).sum(-1);agree=(actual.argmax(-1)==ref.argmax(-1)).float()
  records.append(dict(ids=case,max_abs_error=float(delta.max()),mean_abs_error=float(delta.mean()),mean_kl_nats=float(kl.mean()),top1_agreement=float(agree.mean()),backend_timing=[line for line in r.stderr.splitlines() if line.startswith('CDM:')]))
  maximum=max(maximum,float(delta.max()));agreements.extend(agree.tolist());kls.extend(kl.tolist())
 result=dict(gguf=a.gguf,mode='incremental KV, teacher-forced prefixes',threads=a.threads,cases=records,max_abs_error=maximum,mean_kl_nats=float(np.mean(kls)),top1_agreement=float(np.mean(agreements)),acceptance_thresholds=dict(max_abs_error=.5,mean_kl_nats=.01,top1_agreement=.9));result['passed']=maximum<=.5 and result['mean_kl_nats']<=.01 and result['top1_agreement']>=.9;print(json.dumps(result,indent=2))
 if a.output:pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
