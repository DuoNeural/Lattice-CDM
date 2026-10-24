#!/usr/bin/env python3
"""Compare our frozen reference against locally reviewed original architecture AST.
Only function/class definitions are compiled; original imports/top-level code do not run.
"""
import argparse, ast, json, math, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from cdm_tools import load, forward, PRESETS

def source_model(path,cfg,state,v3=False):
 tree=ast.parse(pathlib.Path(path).read_text()); names={'_sqrt_c','clip_to_ball','exp_map_0','log_map_0','poincare_dist','DualGeometryCDMRouter','ExpertFFN','CDMMoEBlock','CDMLayer','CDMMoELM'}
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 ns={'torch':torch,'nn':nn,'F':F,'math':math}
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 kwargs=dict(vocab_size=cfg['vocab'],d_model=cfg['dim'],n_layers=cfg['layers'],n_heads=cfg['heads'],K=cfg['slots'],num_experts=cfg['experts'],d_ff=cfg['ff'],max_seq=cfg['context'],dropout=0.,geometry='poincare',curvature=cfg.get('curvature',1.4427),use_embed_norm='embed_norm.weight' in state)
 if v3:kwargs.update(learn_curvature=True,use_telepathy=True,beta_tele=cfg['beta_tele'])
 model=ns['CDMMoELM'](**kwargs).eval()
 model.load_state_dict({k:v for k,v in state.items() if not k.endswith('ball_slots')},strict=True)
 return model

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--preset',choices=PRESETS,required=True);p.add_argument('--source',required=True);p.add_argument('--ids',default='1,2,3,4');p.add_argument('--output');a=p.parse_args();torch.set_num_threads(4)
 s,cfg=load(a.checkpoint,PRESETS[a.preset]);v3='beta_tele' in cfg;model=source_model(a.source,cfg,s,v3=v3);ids=torch.tensor([int(x) for x in a.ids.split(',')])
 with torch.inference_mode():
  kwargs={'training':False,'allow_crystallize':False}
  if v3:kwargs['telepathy_scale']=cfg.get('telepathy_scale',1.0)
  h,_,_=model(ids[None],**kwargs);actual=model.lm_head(h)[0];ours=forward(s,cfg,ids)
 delta=(actual-ours).abs();result={'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),'top1_agreement':float((actual.argmax(-1)==ours.argmax(-1)).float().mean()),'allclose_atol_1e-4_rtol_1e-4':bool(torch.allclose(actual,ours,atol=1e-4,rtol=1e-4))};print(json.dumps(result,indent=2))
 if a.output:pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
 if not result['allclose_atol_1e-4_rtol_1e-4']:raise SystemExit(1)
if __name__=='__main__':main()
