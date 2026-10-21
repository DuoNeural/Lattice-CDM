"""Synthetic v3 source parity with nonzero telepathy and nonuniform curvature."""
import ast,json,math,pathlib,tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from cdm_tools import load,forward,export

def main():
 torch.set_num_threads(2);torch.manual_seed(42)
 source=str(pathlib.Path(__file__).parent/'research/cdm_lm_distill_v3.py')
 names={'_sqrt_c','clip_to_ball','exp_map_0','log_map_0','poincare_dist','DualGeometryCDMRouter','ExpertFFN','CDMMoEBlock','CDMLayer','CDMMoELM'}
 tree=ast.parse(pathlib.Path(source).read_text());ns=dict(torch=torch,nn=nn,F=F,math=math)
 exec(compile(ast.Module([n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],[]),source,'exec'),ns)
 model=ns['CDMMoELM'](vocab_size=64,d_model=16,n_layers=3,n_heads=4,K=8,num_experts=2,d_ff=24,max_seq=16,dropout=0.,beta_tele=.7).eval()
 with torch.no_grad():
  for i,l in enumerate(model.layers):
   l.moe.router.theta_c.fill_(-1+i)
   if l.moe.router.W_tele is not None:l.moe.router.W_tele.normal_(0,.6)
 with tempfile.TemporaryDirectory() as d:
  p=pathlib.Path(d)/'v3.pt';torch.save({'model':model.state_dict()},p)
  s,cfg=load(p,dict(heads=4,geometry='poincare',c_min=.25,c_max=4.,beta_tele=.7,telepathy_scale=.8))
  torch.save({'model':model.state_dict()},pathlib.Path(__file__).parent/'models/synthetic-v3.pt')
  export(s,cfg,str(pathlib.Path(__file__).parent/'models/synthetic-v3-f32.gguf'))
  (pathlib.Path(__file__).parent/'models/synthetic-v3-config.json').write_text(json.dumps(dict(heads=4,geometry='poincare',c_min=.25,c_max=4.,beta_tele=.7,telepathy_scale=.8)))
  ids=torch.tensor([2,5,8,12,2])
  with torch.inference_mode():
   h,_,_=model(ids[None],telepathy_scale=.8,training=False,allow_crystallize=False);expected=model.lm_head(h)[0];actual=forward(s,cfg,ids)
  err=(expected-actual).abs();result=dict(max_abs_error=float(err.max()),mean_abs_error=float(err.mean()),curvatures=cfg['curvatures'],betas=cfg['betas'])
  print(json.dumps(result,indent=2));assert torch.allclose(actual,expected,atol=1e-4,rtol=1e-4)
if __name__=='__main__':main()
