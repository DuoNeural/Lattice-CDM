#!/usr/bin/env python3
"""Selective Q8_0, Q6_K, or mixed Q4_K quantization with frozen FP32 geometry."""
import argparse,json,pathlib,re,sys,subprocess,tempfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'llama.cpp/gguf-py'))
import gguf

def quantize(source,dest,quant_type="q8_0",threads=4):
 source=pathlib.Path(source);dest=pathlib.Path(dest)
 if source.resolve()==dest.resolve():raise ValueError('Input and output must differ')
 r=gguf.GGUFReader(source)
 if quant_type!='q8_0':return quantize_native(source,dest,r,quant_type,threads)
 if r.get_field('general.architecture').contents()!='cdm':raise ValueError('Expected CDM architecture')
 w=gguf.GGUFWriter(dest,'cdm',use_temp_file=True)
 for f in r.fields.values():
  if f.name=='general.architecture' or f.name.startswith('GGUF.') or f.name in ['general.file_type','cdm.quantization.policy']:continue
  typ=f.types[0];w.add_key_value(f.name,f.contents(),typ,sub_type=f.types[-1] if typ==gguf.GGUFValueType.ARRAY else None)
 w.add_uint32('general.file_type',int(gguf.LlamaFileType.MOSTLY_Q8_0));w.add_string('cdm.quantization.policy','Q8_0 embeddings, attention and expert matrices; F32 positions, norms, routing geometry, telepathy and mapping')
 counts={};names=[]
 for t in r.tensors:
  eligible=t.name in ['token_embd.weight','output.weight'] or bool(re.fullmatch(r'blk\.\d+\.(attn_qkv|attn_output|cdm_expert_\d+_(gate|up|down))\.weight',t.name))
  if t.tensor_type!=gguf.GGMLQuantizationType.F32:raise ValueError('Input must be all F32')
  if eligible:
   if t.data.shape[-1]%32:raise ValueError('Q8_0 requires aligned matrix rows: '+t.name)
   data=gguf.quantize(t.data,gguf.GGMLQuantizationType.Q8_0);w.add_tensor(t.name,data,raw_dtype=gguf.GGMLQuantizationType.Q8_0);names.append(t.name)
  else:w.add_tensor(t.name,t.data)
 w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
 check=gguf.GGUFReader(dest)
 for t in check.tensors:
  wanted=gguf.GGMLQuantizationType.Q8_0 if t.name in names else gguf.GGMLQuantizationType.F32
  if t.tensor_type!=wanted:raise ValueError('Quantization type mismatch: '+t.name)
  counts[t.tensor_type.name]=counts.get(t.tensor_type.name,0)+1
 result=dict(input=str(source),output=str(dest),input_bytes=source.stat().st_size,output_bytes=dest.stat().st_size,tensor_types=counts,quantized_tensor_names=names)
 dest.with_suffix(dest.suffix+'.quantization.json').write_text(json.dumps(result,indent=2)+'\n');return result
def quantize_native(source,dest,reader,kind,threads):
 if kind not in ('q6_k','q4_k_m','q4_k_s'):raise ValueError('Unsupported quantization type')
 if reader.get_field('general.architecture').contents()!='cdm':raise ValueError('Expected CDM architecture')
 expected={};rules=[]
 for t in reader.tensors:
  if t.tensor_type!=gguf.GGMLQuantizationType.F32:raise ValueError('Input must be all F32')
  eligible=t.name in ['token_embd.weight','output.weight'] or bool(re.fullmatch(r'blk\.\d+\.(attn_qkv|attn_output|cdm_expert_\d+_(gate|up|down))\.weight',t.name))
  typ='F32'
  if eligible:
   if t.data.shape[-1]%256:raise ValueError('K-quant requires 256-aligned matrix rows: '+t.name)
   typ='Q4_K' if kind=='q4_k_s' else ('Q6_K' if kind=='q6_k' or t.name in ['token_embd.weight','output.weight'] or '_down.weight' in t.name else 'Q4_K')
  expected[t.name]=typ;rules.append('^'+re.escape(t.name)+'$='+typ.lower())
 policy=('Q4_K all large matrices' if kind=='q4_k_s' else ('Q6_K all large matrices' if kind=='q6_k' else 'Q4_K attention and expert gate/up; Q6_K tied embeddings/output and expert down'))+'; F32 positions, norms, routing geometry, telepathy and mapping'
 with tempfile.TemporaryDirectory() as tmp:
  rulesfile=pathlib.Path(tmp)/'tensor-types.txt';rulesfile.write_text('\n'.join(rules)+'\n')
  args=[str(ROOT/'llama.cpp/build/bin/llama-quantize'),'--tensor-type-file',str(rulesfile),'--override-kv','cdm.quantization.policy=str:'+('Q4_K all large matrices; F32 geometry/norm/position' if kind=='q4_k_s' else ('Q6_K large matrices; F32 geometry/norm/position' if kind=='q6_k' else 'Q4_K attn/gate/up; Q6_K embedding/down; F32 geometry/norm/position')),str(source),str(dest),kind.upper(),str(threads)]
  proc=subprocess.run(args,capture_output=True,text=True,timeout=600)
  pathlib.Path(str(dest)+'.quantize.log').write_text(proc.stdout+proc.stderr)
  if proc.returncode:raise RuntimeError('Native quantizer failed: '+proc.stderr[-5000:])
 counts={}
 check=gguf.GGUFReader(dest)
 for t in check.tensors:
  if t.tensor_type.name!=expected[t.name]:raise ValueError('Quantization type mismatch: '+t.name+' wanted '+expected[t.name]+' got '+t.tensor_type.name)
  counts[t.tensor_type.name]=counts.get(t.tensor_type.name,0)+1
 result=dict(input=str(source),output=str(dest),type=kind,policy=policy,input_bytes=source.stat().st_size,output_bytes=dest.stat().st_size,tensor_types=counts,tensor_type_manifest=expected)
 pathlib.Path(str(dest)+'.quantization.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('output');p.add_argument('--type',choices=['q8_0','q6_k','q4_k_m','q4_k_s'],default='q8_0');p.add_argument('--threads',type=int,default=4);a=p.parse_args();print(json.dumps(quantize(a.input,a.output,a.type,a.threads),indent=2))
