#!/usr/bin/env python3
"""Deterministic CDM numerical probes; no task-quality or perplexity claim.

Eight natural-text/code/structured/edge prompts are teacher-forced through 64
positions each, using immutable teacher-tokenizer revisions. All quantizations
share the same thresholds; failures are recorded without changing tolerances.
GPU runs require native full-offload log evidence, not merely a requested flag.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np
import torch
from cdm_tools import ROOT, PRESETS, forward, load

MODELS = {
    'sub17m': 'cdm_sublattice_v1_poincare_final_step0100000.pt',
    'lattice107m': 'pytorch_model.pt',
}
THRESHOLDS = {
    'f32': {'atol': 1e-3, 'rtol': 1e-4, 'require_allclose': True},
    'quantized': {'max_abs_error': .5, 'mean_kl_nats': .01, 'top1_agreement': .9},
}
PROBES = {
    'prose': 'The observatory opened before dawn. A technician recorded the temperature, checked the old telescope, and watched a pale line of light move across the valley. Every measurement went into a paper notebook because the network was unreliable. The team compared observations from three nights before deciding whether the signal was real. ',
    'instruction': 'Explain how to organize a small research project. State the question, list assumptions, record reproducible measurements, compare a baseline, and describe limitations. Use concrete examples and distinguish observations from speculation. A useful experiment should be understandable to another person who has never seen the original notebook. ',
    'python': 'def merge_sorted(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    return result + left[i:] + right[j:]\n',
    'json': '{"experiment":"crystal-routing","version":3,"enabled":true,"layers":[{"id":0,"curvature":0.25,"slots":[0,1,2,3]},{"id":1,"curvature":1.4427,"slots":[4,5,6,7]}],"notes":null,"tags":["baseline","repeatable","offline"],"metrics":{"loss":1.2345,"tokens":512,"seed":1994}}\n',
    'numbers': 'Compute carefully: 17 + 28 = 45; 144 / 12 = 12; 0.125 * 8 = 1. Dates: 1994-10-21 and 2026-10-21. Hexadecimal: 0x00, 0x7f, 0xff. Scientific notation: 1e-12, 3.14159265, -2.5e+4. Compare 1024 bytes, 2048 bytes, and 4096 bytes. ',
    'unicode_whitespace': 'English: memory and geometry. Français: mémoire et géométrie. Español: información. Ελληνικά: δοκιμή. 日本語: テスト。中文：你好。Emoji: 💎 🛰️.\n\nTabs:\talpha\tbeta\nSpaces:    one     two.\nQuoted: "a\\nb" and a literal backslash \\. ',
    'repetition_boundaries': 'zero zero zero one one two two two three.\n' + 'a b a b a b 0 1 0 1\n' * 12 + '\nEND.\n',
    'conversation': 'I am debugging a small local application. The program reads a JSON configuration and returns a sorted list of experiment names. Explain how to handle an empty file, a missing key, malformed JSON, and duplicate names. Include a concise example and explain how to reproduce the issue without changing production data. ',
}


def pinned_cases(model, positions):
    from transformers import AutoTokenizer
    provenance = ROOT/'models'/f'{model}-f32-text.gguf.tokenizer.json'
    info = json.loads(provenance.read_text())
    revision = info['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Tokenizer revision must be an immutable 40-character commit')
    tokenizer = AutoTokenizer.from_pretrained(info['tokenizer'], revision=revision,
                                              trust_remote_code=False, local_files_only=True)
    cases = []
    for name, text in PROBES.items():
        rendered = text * 8
        chat = model == 'sub17m' and name == 'conversation'
        if chat:
            rendered = tokenizer.apply_chat_template([
                {'role':'system','content':'You are a careful research assistant.'},
                {'role':'user','content':rendered}], tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) < positions: raise ValueError('Probe unexpectedly too short: '+name)
        # Keep complete ChatML role/open-assistant framing in the conversation case.
        ids = ids[:positions] if not chat else ids[:positions-5]+ids[-5:]
        cases.append({'name':name,'chat_template':chat,'ids':ids,
                      'rendered_text_sha256':hashlib.sha256(rendered.encode()).hexdigest()})
    return info, cases


def metrics(reference, actual, ids):
    if actual.shape != reference.shape: raise ValueError('Backend logit shape mismatch')
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite: return {'finite':False,'positions':len(reference),'passed':False}
    # Double precision statistics and logsumexp prevent overflow/cancellation in KL.
    a = torch.from_numpy(actual).double()
    r = torch.from_numpy(reference).double()
    delta = a-r
    rl=r.log_softmax(-1); al=a.log_softmax(-1)
    kl = (rl.exp()*(rl-al)).sum(-1)
    targets=torch.tensor(ids[1:],dtype=torch.long)
    rows=torch.arange(len(targets))
    ref_nll=float(-rl[:-1][rows,targets].mean())
    actual_nll=float(-al[:-1][rows,targets].mean())
    return {'finite':True,'positions':len(reference),'vocab_size':reference.shape[-1],
            'max_abs_error':float(delta.abs().max()),
            'mean_abs_error':float(delta.abs().mean()),
            'rms_error':float(delta.square().mean().sqrt()),
            'mean_kl_nats':float(kl.mean()),'max_kl_nats':float(kl.max()),
            'top1_agreement':float((a.argmax(-1)==r.argmax(-1)).double().mean()),
            'scored_next_tokens':len(targets),'reference_next_token_nll':ref_nll,
            'native_next_token_nll':actual_nll,'delta_next_token_nll':actual_nll-ref_nll,
            'relative_probe_perplexity':math.exp(actual_nll-ref_nll),
            'allclose_f32':bool(np.allclose(actual,reference,atol=THRESHOLDS['f32']['atol'],rtol=THRESHOLDS['f32']['rtol']))}


def aggregate(records, kind):
    if any(not r.get('finite',False) for r in records): return {'finite':False,'passed':False}
    positions = sum(r['positions'] for r in records)
    result = {'finite':True,'positions':positions,
              'max_abs_error':max(r['max_abs_error'] for r in records),
              'max_kl_nats':max(r['max_kl_nats'] for r in records),
              'allclose_f32':all(r['allclose_f32'] for r in records)}
    for key in ['mean_abs_error','mean_kl_nats','top1_agreement']:
        result[key] = sum(r[key]*r['positions'] for r in records)/positions
    result['rms_error'] = math.sqrt(sum(r['rms_error']**2*r['positions'] for r in records)/positions)
    scored=sum(r['scored_next_tokens'] for r in records)
    result['scored_next_tokens']=scored
    for key in ['reference_next_token_nll','native_next_token_nll']:
        result[key]=sum(r[key]*r['scored_next_tokens'] for r in records)/scored
    result['delta_next_token_nll']=result['native_next_token_nll']-result['reference_next_token_nll']
    result['relative_probe_perplexity']=math.exp(result['delta_next_token_nll'])
    if kind == 'f32': result['passed'] = result['allclose_f32']
    else:
        t=THRESHOLDS['quantized']
        result['passed'] = result['max_abs_error']<=t['max_abs_error'] and result['mean_kl_nats']<=t['mean_kl_nats'] and result['top1_agreement']>=t['top1_agreement']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',type=Path,default=ROOT/'cdm-logits')
    p.add_argument('--gpu-layers',type=int,default=0)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--formats',nargs='+',default=['f32','q8_0','q6_k','q4_k_m','q4_k_s'])
    p.add_argument('--modes',nargs='+',choices=['prefill','incremental'],default=['prefill','incremental'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--positions-per-case',type=int,default=64)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--artifact',action='append',default=[],metavar='MODEL:FORMAT:PATH',help='Override an artifact path')
    a=p.parse_args()
    if not 8<=a.positions_per_case<=512: p.error('positions-per-case must be 8..512')
    if a.gpu_layers<0: p.error('gpu-layers must be nonnegative; 99 requests full offload')
    overrides={}
    for override in a.artifact:
        model,fmt,path=override.split(':',2);overrides[(model,fmt)]=Path(path)
    torch.set_num_threads(a.threads)
    result={'description':'Authored deterministic numerical corpus probes with teacher-forced next-token NLL; no heldout/benchmark/task-quality claim',
            'runner':str(a.runner.resolve()),'runner_sha256':hashlib.sha256(a.runner.read_bytes()).hexdigest(),
            'router_branch':'unrolled' if os.environ.get('CDM_UNROLLED_ROUTER')=='1' else 'vectorized',
            'backend_source_sha256':hashlib.sha256((ROOT/'llama.cpp/src/models/cdm.cpp').read_bytes()).hexdigest(),
            'requested_gpu_layers':a.gpu_layers,
            'threads':a.threads,'precision_environment':{k:os.environ.get(k) for k in ['GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_FORCE_MMQ','GGML_CUDA_FORCE_CUBLAS','CUDA_VISIBLE_DEVICES']},'thresholds':THRESHOLDS,'threshold_scope':'aggregate over all 8 probes, same quantized thresholds for every format',
            'models':{},'passed':True}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temporary=a.output.with_suffix(a.output.suffix+'.tmp')
        temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        temporary.replace(a.output)
    for model in a.models:
        provenance,cases=pinned_cases(model,a.positions_per_case)
        s,cfg=load(ROOT/'models'/MODELS[model],PRESETS[model])
        references=[]
        for case in cases:
            with torch.inference_mode(): references.append(forward(s,cfg,torch.tensor(case['ids'])).numpy())
        del s
        mr={'tokenizer':provenance,'probes':cases,'positions':sum(len(c['ids']) for c in cases),'runs':[]}
        result['models'][model]=mr
        for fmt in a.formats:
            artifact=overrides.get((model,fmt),ROOT/'models'/f'{model}-{fmt}-text.gguf')
            if not artifact.exists():
                mr['runs'].append({'format':fmt,'artifact':str(artifact),'error':'missing artifact','passed':False})
                result['passed']=False;save();continue
            for mode in a.modes:
                run={'format':fmt,'artifact':str(artifact),'mode':mode,'cases':[]}
                mr['runs'].append(run)
                for case,reference in zip(cases,references):
                    env=dict(os.environ,CDM_THREADS=str(a.threads),CDM_GPU_LAYERS=str(a.gpu_layers),
                             CDM_INCREMENTAL='1' if mode=='incremental' else '0',CDM_REPEATS='1')
                    env.pop('CDM_PROMPT',None)
                    record={'name':case['name']}
                    try:
                        with tempfile.TemporaryDirectory() as folder:
                            out=Path(folder)/'logits.f32'
                            proc=subprocess.run([str(a.runner.resolve()),str(artifact.resolve()),str(out),*map(str,case['ids'])],
                                                env=env,capture_output=True,text=True,timeout=600)
                            record['backend_stderr']=proc.stderr
                            if proc.returncode: raise RuntimeError(f'Backend exit {proc.returncode}')
                            actual=np.fromfile(out,dtype=np.float32)
                            if actual.size!=reference.size:raise ValueError('Incorrect output file size')
                            record.update(metrics(reference,actual.reshape(reference.shape),case['ids']))
                            offload=re.findall(r'offloaded\s+(\d+)/(\d+)\s+layers',proc.stderr)
                            record['offloaded_layers']=[list(map(int,pair)) for pair in offload]
                            record['full_gpu_offload_verified']=any(int(n)==int(total) and int(total)>0 for n,total in offload)
                            if a.gpu_layers>0 and not record['full_gpu_offload_verified']:
                                record['finite']=False;record['error']='Requested GPU run lacks native full-offload evidence'
                    except Exception as exc:
                        record.update(finite=False,passed=False,error=str(exc))
                    run['cases'].append(record)
                run.update(aggregate(run['cases'],'f32' if fmt=='f32' else 'quantized'))
                result['passed'] &= run['passed']
                print(json.dumps({'model':model,'format':fmt,'mode':mode,**{k:v for k,v in run.items() if k not in ['cases','artifact','format','mode']}}),flush=True)
                save()
    save()
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
