#!/usr/bin/env python3
"""Auditable frozen CDM inference and GGUF export. Never imports checkpoint code."""
import argparse, json, math, pathlib, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'llama.cpp/gguf-py'))
PRESETS = {
 'sub17m': dict(heads=4, geometry='poincare', curvature=1.4427, source='DuoNeural/Sub-Lattice-17M-Poincare-Instruct README, published v1'),
 'lattice107m': dict(heads=8, geometry='poincare', curvature=1.4427, source='DuoNeural/Lattice-1-107M-Poincare-Preview README, published v1'),
}

def expmap(v,c):
 n=v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
 return torch.tanh((math.sqrt(c)*n).clamp_max(8))/(math.sqrt(c)*n)*v

def clip(v,c):
 n=v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
 return v*torch.minimum(torch.ones_like(n), ((1-1e-5)/math.sqrt(c))/n)

def distance(x,s,c):
 x=x.unsqueeze(-2); s=s.unsqueeze(0)
 cx=(c*x.square().sum(-1)).clamp(0,1-1e-5)
 cs=(c*s.square().sum(-1)).clamp(0,1-1e-5)
 arg=1+2*c*(x-s).square().sum(-1)/((1-cx)*(1-cs)).clamp_min(1e-5)
 return (2/math.sqrt(c))*torch.acosh(arg.clamp_min(1+1e-5))


def validate_checkpoint(s, config):
 """Fail closed before deriving metadata or transforming any checkpoint tensor."""
 def fail(message): raise ValueError(message)
 if config.get('geometry') != 'poincare':
  fail('This prototype requires explicit geometry=poincare; other geometries are unsupported')
 heads=config.get('heads')
 if type(heads) is not int or heads<=0: fail('heads must be a positive integer')
 for key,value in s.items():
  if not isinstance(key,str) or not isinstance(value,torch.Tensor):fail('Expected string tensor names and tensors')
  if value.is_complex() or value.is_quantized:fail('Unsupported tensor dtype: '+key)
  if value.is_floating_point() and not torch.isfinite(value).all():fail('Nonfinite tensor: '+key)
 expected=set();training=set()
 def shape(key,wanted):
  expected.add(key)
  if key not in s:fail('Missing required tensor: '+key)
  t=s[key]
  if tuple(t.shape)!=tuple(wanted):fail(f'Invalid shape for {key}: {tuple(t.shape)} != {tuple(wanted)}')
  if not t.is_floating_point():fail('Expected floating-point tensor: '+key)
 def matrix_dims(key):
  if key not in s or s[key].ndim!=2 or min(s[key].shape)<=0:fail('Expected nonempty matrix: '+key)
  return tuple(s[key].shape)
 vocab,d=matrix_dims('tok_embed.weight');context,pd=matrix_dims('pos_embed.weight')
 if pd!=d or d%heads:fail('Invalid position dimension or attention head count')
 shape('tok_embed.weight',(vocab,d));shape('lm_head.weight',(vocab,d));shape('pos_embed.weight',(context,d))
 if not torch.equal(s['tok_embed.weight'],s['lm_head.weight']):fail('Untied output head is unsupported')
 for name in ['final_norm']+(['embed_norm'] if any(k.startswith('embed_norm.') for k in s) else []):
  shape(name+'.weight',(d,));shape(name+'.bias',(d,))
 layers=set()
 for key in s:
  if key.startswith('layers.'):
   m=re.match(r'^layers\.(0|[1-9][0-9]*)\.',key)
   if not m:fail('Malformed layer name: '+key)
   layers.add(int(m.group(1)))
 if not layers or sorted(layers)!=list(range(len(layers))):fail('Expected nonempty contiguous layer indexes')
 representations=set();ks=set();es=set();fs=set()
 for i in sorted(layers):
  p=f'layers.{i}.';r=p+'moe.router.'
  for name in ['attn_norm','ffn_norm']:
   shape(p+name+'.weight',(d,));shape(p+name+'.bias',(d,))
  shape(p+'attn.in_proj_weight',(3*d,d));shape(p+'attn.in_proj_bias',(3*d,))
  shape(p+'attn.out_proj.weight',(d,d));shape(p+'attn.out_proj.bias',(d,))
  present=[name for name in ['slots','slots_tangent'] if r+name in s]
  if len(present)!=1:fail('Expected exactly one slot representation in layer '+str(i))
  representation=present[0];representations.add(representation)
  k,sd=matrix_dims(r+representation)
  if sd!=d:fail('Slot dimension mismatch')
  ks.add(k);shape(r+representation,(k,d));shape(r+'log_temp',())
  expert_ids=set()
  for key in s:
   if key.startswith(p+'moe.experts.'):
    m=re.match(re.escape(p)+r'moe\.experts\.(0|[1-9][0-9]*)\.',key)
    if not m:fail('Malformed expert name: '+key)
    expert_ids.add(int(m.group(1)))
  if not expert_ids or sorted(expert_ids)!=list(range(len(expert_ids))):fail('Noncontiguous or empty experts')
  e=len(expert_ids);es.add(e)
  ff,fd=matrix_dims(p+'moe.experts.0.gate.weight')
  if fd!=d:fail('Expert dimension mismatch')
  fs.add(ff)
  for j in range(e):
   ep=p+f'moe.experts.{j}.'
   shape(ep+'gate.weight',(ff,d));shape(ep+'up.weight',(ff,d));shape(ep+'down.weight',(d,ff))
  mk=r+'slot_to_expert';expected.add(mk)
  if mk not in s:fail('Missing slot-to-expert mapping')
  mapping=s[mk]
  if mapping.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8) or tuple(mapping.shape)!=(k,) or mapping.min()<0 or mapping.max()>=e:
   fail('Invalid integer slot-to-expert mapping')
  if r+'theta_c' in s:
   if representation!='slots_tangent':fail('Learned curvature requires tangent slot version')
   shape(r+'theta_c',())
   low,high=config.get('c_min'),config.get('c_max')
   if not isinstance(low,(int,float)) or not isinstance(high,(int,float)) or not math.isfinite(low) or not math.isfinite(high) or not 0<low<high:
    fail('Learned curvature requires finite 0 < c_min < c_max')
  else:
   c=config.get('curvature')
   if not isinstance(c,(int,float)) or not math.isfinite(c) or c<=0:fail('Fixed curvature must be explicit, finite and positive')
  if r+'W_tele' in s:
   if representation!='slots_tangent' or i==0:fail('Telepathy requires v3 and a predecessor layer')
   shape(r+'W_tele',(k,k))
   for name in ['beta_tele','telepathy_scale']:
    value=config.get(name)
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:fail('Telepathy requires explicit finite nonnegative '+name)
  training.add(p+'_causal')
  for name in ['alpha','tau_star','slot_cum_weight','prev_tau_per_slot','prev_tau_max']:
   training.add(r+name)
 if len(representations)!=1 or len(ks)!=1 or len(es)!=1 or len(fs)!=1:fail('Mixed layer versions or inconsistent slot/expert/FFN dimensions unsupported')
 unknown=set(s)-expected-training
 if unknown:fail('Unknown tensors would be silently ignored: '+', '.join(sorted(unknown)))


def load(path,config):
 ck=torch.load(path,map_location='cpu',weights_only=True)
 state=ck.get('model',ck.get('state_dict',ck))
 if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):
  raise ValueError('Expected tensor-only state dictionary in checkpoint model/state_dict')
 validate_checkpoint(state,config)
 s={k:v.detach().float() if v.is_floating_point() else v.detach() for k,v in state.items()}
 if any(v.is_floating_point() and not torch.isfinite(v).all() for v in s.values()):raise ValueError('Tensor overflows float32 inference representation')
 d=s['tok_embed.weight'].shape[1]; layers=sorted({int(k.split('.')[1]) for k in s if k.startswith('layers.')})
 if layers!=list(range(len(layers))): raise ValueError('Non-contiguous layer indexes')
 cfg=dict(config);cfg.update(dim=d,layers=len(layers),vocab=s['tok_embed.weight'].shape[0],context=s['pos_embed.weight'].shape[0])
 if d%cfg['heads']:raise ValueError('Heads do not divide embedding dimension')
 if not torch.equal(s['tok_embed.weight'],s['lm_head.weight']):raise ValueError('Untied output head is unsupported')
 cfg['curvatures']=[];cfg['temperatures']=[];cfg['betas']=[]
 for i in layers:
  p=f'layers.{i}.moe.router.'
  if p+'theta_c' in s:
   if 'c_min' not in cfg or 'c_max' not in cfg:raise ValueError('Learned curvature requires explicit c_min/c_max from training configuration')
   c=float(cfg['c_min']+(cfg['c_max']-cfg['c_min'])*s[p+'theta_c'].sigmoid())
  else:
   if 'curvature' not in cfg:raise ValueError('Fixed curvature absent: supply explicit curvature')
   c=float(cfg['curvature'])
  if not math.isfinite(c) or c<=0:raise ValueError('Curvature must be finite and positive')
  cfg['curvatures'].append(c);cfg['temperatures'].append(float(s[p+'log_temp'].clamp(math.log(.5),math.log(20)).exp()))
  slots=s.get(p+'slots',None)
  if slots is None:slots=expmap(s[p+'slots_tangent'],c)
  s[p+'ball_slots']=clip(slots,c)
  if p+'W_tele' in s:
   if 'beta_tele' not in cfg or 'telepathy_scale' not in cfg:raise ValueError('Telepathy requires explicit beta_tele and telepathy_scale')
   beta=cfg['beta_tele']*cfg['telepathy_scale']
  else:beta=0.
  cfg['betas'].append(beta)
 cfg['slots']=s['layers.0.moe.router.ball_slots'].shape[0]
 cfg['experts']=len([k for k in s if k.startswith('layers.0.moe.experts.') and k.endswith('gate.weight')])
 cfg['ff']=s['layers.0.moe.experts.0.gate.weight'].shape[0]
 return s,cfg

def norm(x,s,p):return F.layer_norm(x,(x.shape[-1],),s[p+'.weight'],s[p+'.bias'],1e-5)

def forward(s,cfg,ids):
 if ids.ndim!=1 or not 0<len(ids)<=cfg['context']:raise ValueError('Expected nonempty 1D token IDs within context length')
 if ids.min()<0 or ids.max()>=cfg['vocab']:raise ValueError('Token ID outside vocabulary')
 x=F.embedding(ids,s['tok_embed.weight'])+s['pos_embed.weight'][:len(ids)]
 if 'embed_norm.weight' in s:x=norm(x,s,'embed_norm')
 prev=None
 for i in range(cfg['layers']):
  p=f'layers.{i}';h=norm(x,s,p+'.attn_norm');q,k,v=F.linear(h,s[p+'.attn.in_proj_weight'],s[p+'.attn.in_proj_bias']).chunk(3,-1)
  def split(t):return t.reshape(len(ids),cfg['heads'],-1).transpose(0,1).unsqueeze(0)
  a=F.scaled_dot_product_attention(split(q),split(k),split(v),is_causal=True).squeeze(0).transpose(0,1).reshape(len(ids),-1)
  x=x+F.linear(a,s[p+'.attn.out_proj.weight'],s[p+'.attn.out_proj.bias']);h=norm(x,s,p+'.ffn_norm')
  r=p+'.moe.router.';c=cfg['curvatures'][i]
  z=clip(expmap(F.normalize(h,dim=-1)*.3,c),c)
  logits=-distance(z,s[r+'ball_slots'],c)*cfg['temperatures'][i]
  if prev is not None and cfg['betas'][i]:logits=logits+cfg['betas'][i]*(prev@s[r+'W_tele'])
  prev=logits.softmax(-1);ew=torch.zeros(len(ids),cfg['experts'])
  for j,e in enumerate(s[r+'slot_to_expert'].tolist()):ew[:,e]+=prev[:,j]
  out=torch.zeros_like(h)
  for e in range(cfg['experts']):
   ep=p+f'.moe.experts.{e}.'
   out+=ew[:,e,None]*F.linear(F.silu(F.linear(h,s[ep+'gate.weight']))*F.linear(h,s[ep+'up.weight']),s[ep+'down.weight'])
  x=x+out
 return F.linear(norm(x,s,'final_norm'),s['lm_head.weight'])

def export(s,cfg,out):
 import gguf
 w=gguf.GGUFWriter(out,'cdm')
 w.add_string('tokenizer.ggml.model','none');w.add_name('DuoNeural CDM frozen inference prototype');w.add_uint32('cdm.context_length',cfg['context']);w.add_uint32('cdm.embedding_length',cfg['dim']);w.add_uint32('cdm.block_count',cfg['layers']);w.add_uint32('cdm.feed_forward_length',cfg['ff']);w.add_uint32('cdm.attention.head_count',cfg['heads']);w.add_float32('cdm.attention.layer_norm_epsilon',1e-5)
 w.add_uint32('cdm.vocab_size',cfg['vocab']);w.add_uint32('cdm.cdm.slot_count',cfg['slots']);w.add_uint32('cdm.cdm.expert_count',cfg['experts'])
 for key,val in [('curvature',cfg['curvatures']),('temperature',cfg['temperatures']),('telepathy_beta',cfg['betas'])]:w.add_array('cdm.cdm.'+key+'_per_layer',val)
 w.add_string('cdm.source.configuration',json.dumps(cfg,sort_keys=True))
 def add(name,t):w.add_tensor(name,t.contiguous().numpy().astype(np.float32))
 add('token_embd.weight',s['tok_embed.weight']);add('position_embd.weight',s['pos_embed.weight'])
 for src,dst in [('embed_norm','token_embd_norm'),('final_norm','output_norm')]:
  for suffix in ['weight','bias']:
   key=src+'.'+suffix
   if key in s:add(dst+'.'+suffix,s[key])
 for i in range(cfg['layers']):
  p=f'layers.{i}.';b=f'blk.{i}.'
  for src,dst in [('attn_norm','attn_norm'),('ffn_norm','ffn_norm'),('attn.out_proj','attn_output')]:
   for suffix in ['weight','bias']:add(b+dst+'.'+suffix,s[p+src+'.'+suffix])
  add(b+'attn_qkv.weight',s[p+'attn.in_proj_weight']);add(b+'attn_qkv.bias',s[p+'attn.in_proj_bias'])
  r=p+'moe.router.';add(b+'cdm_slots.weight',s[r+'ball_slots'])
  add(b+'cdm_tele.weight',s.get(r+'W_tele',torch.zeros(cfg['slots'],cfg['slots'])).T)
  mapping=s[r+'slot_to_expert']
  if mapping.shape!=(cfg['slots'],) or mapping.min()<0 or mapping.max()>=cfg['experts']:raise ValueError('Invalid slot-to-expert mapping')
  add(b+'cdm_slot_expert.weight',F.one_hot(mapping,num_classes=cfg['experts']).T.float())
  for e in range(cfg['experts']):
   for kind in ['gate','up','down']:add(b+f'cdm_expert_{e}_{kind}.weight',s[p+f'moe.experts.{e}.{kind}.weight'])
 w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
 pathlib.Path(str(out)+'.config.json').write_text(json.dumps(cfg,indent=2)+'\n')

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['export','reference','benchmark']);ap.add_argument('checkpoint');ap.add_argument('--preset',choices=PRESETS);ap.add_argument('--config',help='JSON with explicit heads/curvature or learned bounds/telepathy settings');ap.add_argument('--output');ap.add_argument('--ids',default='1,2,3,4');ap.add_argument('--threads',type=int,default=4);ap.add_argument('--repeats',type=int,default=5)
 a=ap.parse_args();torch.set_num_threads(a.threads);config=dict(PRESETS.get(a.preset,{}))
 if a.config:config.update(json.loads(pathlib.Path(a.config).read_text()))
 if 'heads' not in config:ap.error('Supply --preset or --config with explicit heads')
 s,cfg=load(a.checkpoint,config)
 if a.command=='export':
  if not a.output:ap.error('export requires --output')
  export(s,cfg,a.output);print(json.dumps(cfg,indent=2));return
 ids=torch.tensor([int(x) for x in a.ids.split(',')],dtype=torch.long)
 with torch.inference_mode():
  logits=forward(s,cfg,ids)
  if a.command=='reference':
   if a.output:np.save(a.output,logits.numpy())
   print(json.dumps({'shape':list(logits.shape),'top1':logits.argmax(-1).tolist(),'last_top10':logits[-1].topk(10).indices.tolist()}));return
  samples=[]
  for _ in range(a.repeats):
   start=time.perf_counter();forward(s,cfg,ids);samples.append(time.perf_counter()-start)
  print(json.dumps({'mode':'full-prefix forward, no KV cache','ids':len(ids),'threads':a.threads,'seconds':samples,'median_seconds':float(np.median(samples)),'evaluated_tokens_per_second':len(ids)/float(np.median(samples))},indent=2))
if __name__=='__main__':main()
