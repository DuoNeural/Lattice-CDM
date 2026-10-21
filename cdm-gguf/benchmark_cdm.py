#!/usr/bin/env python3
"""CPU FP32 PyTorch full-prefix generation vs libllama cached decoding.
Same teacher-forced IDs and thread count; backend timing excludes loading/reserve.
PyTorch repeats full prefixes because published architecture has no KV cache.
"""
import argparse,json,os,pathlib,re,subprocess,tempfile,time
import numpy as np
import torch
from cdm_tools import ROOT,PRESETS,load,forward

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('gguf');p.add_argument('--preset',choices=PRESETS,required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--repeats',type=int,default=3);p.add_argument('--ids',default='1,2,3,4,17,28,42,198,327,355,281,520,17,28,42,198');p.add_argument('--output');a=p.parse_args();torch.set_num_threads(a.threads);s,cfg=load(a.checkpoint,PRESETS[a.preset]);ids=torch.tensor(list(map(int,a.ids.split(','))))
 py=[];native=[]
 with torch.inference_mode():
  forward(s,cfg,ids)
  for _ in range(a.repeats):
   start=time.perf_counter()
   for j in range(1,len(ids)+1):forward(s,cfg,ids[:j])
   py.append(time.perf_counter()-start)
 with tempfile.TemporaryDirectory() as d:
  for _ in range(a.repeats):
   r=subprocess.run([str(ROOT/'cdm-logits'),a.gguf,str(pathlib.Path(d)/'out.f32'),*map(str,ids.tolist())],env=dict(os.environ,CDM_THREADS=str(a.threads),CDM_INCREMENTAL='1'),capture_output=True,text=True,check=True)
   match=re.search(r'CDM: \d+ tokens, \d+ vocab, ([\d.]+) s',r.stderr)
   if match is None:raise ValueError('Missing native timing')
   native.append(float(match[1]))
 result=dict(threads=a.threads,tokens=len(ids),repeats=a.repeats,pytorch_mode='FP32, uncached full-prefix autoregressive steps',native_mode='incremental KV, F32 cache, excludes load/reservation',pytorch_seconds=py,native_seconds=native,pytorch_tokens_per_second=len(ids)/float(np.median(py)),native_tokens_per_second=len(ids)/float(np.median(native)))
 print(json.dumps(result,indent=2))
 if a.output:pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
