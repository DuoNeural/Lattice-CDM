#!/usr/bin/env python3
"""Measure native CPU/CUDA throughput against a CPU PyTorch reference.

All modes return logits at every input position. Native incremental mode consumes
a fixed teacher-forced sequence; it is not sampled text generation. Only compare
prefill rates directly, because this PyTorch reference has no KV cache.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np
import torch
from cdm_tools import PRESETS, load, forward

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--preset', choices=PRESETS, required=True)
    p.add_argument('--gguf', action='append', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--length', type=int, default=32)
    p.add_argument('--ids', help='Override length with comma-separated IDs')
    p.add_argument('--output', required=True)
    p.add_argument('--runner', type=Path, default=ROOT / 'cdm-logits')
    p.add_argument('--gpu-layers', type=int, default=0)
    p.add_argument('--prefill-count', type=int, default=0,
                   help='In incremental mode, prefill this many IDs before timed decode')
    p.add_argument('--unrolled-router', action='store_true',
                   help='Benchmark the original router graph for comparison')
    a = p.parse_args()
    if a.repeats < 2 or a.threads < 1:
        p.error('Use at least 2 repeats and a positive thread count')
    torch.set_num_threads(a.threads)
    s, cfg = load(a.checkpoint, PRESETS[a.preset])
    ids = ([int(v) for v in a.ids.split(',')] if a.ids else
           [1 + ((i * 7919 + 1994) % (cfg['vocab'] - 1)) for i in range(a.length)])
    tokens = torch.tensor(ids, dtype=torch.long)
    if not 0 <= a.prefill_count < len(ids):
        p.error('prefill-count must be nonnegative and less than sequence length')
    with torch.inference_mode():
        forward(s, cfg, tokens)
        samples = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            forward(s, cfg, tokens)
            samples.append(time.perf_counter() - start)
    result = {'model': a.preset, 'threads': a.threads, 'ids': ids,
              'warmup_runs': 1, 'repeats': a.repeats,
              'hardware': next((line.split(':', 1)[1].strip() for line in
                                Path('/proc/cpuinfo').read_text().splitlines()
                                if line.startswith('model name')), 'unknown'),
              'pytorch_version': torch.__version__,
              'runner': str(a.runner.resolve()), 'gpu_layers_requested': a.gpu_layers,
              'prefill_count': a.prefill_count,
              'router': 'unrolled' if a.unrolled_router else 'vectorized',
              'pytorch_prefill_seconds': samples,
              'pytorch_prefill_tokens_per_second': len(ids) / float(np.median(samples)),
              'native': [],
              'note': 'All-position logits. Native F32 KV. Compare prefill only; '
                      'incremental is teacher-forced cached decoding, not sampled generation.'}
    for filename in a.gguf:
        for incremental in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                env = dict(os.environ, CDM_THREADS=str(a.threads),
                           CDM_REPEATS=str(a.repeats),
                           CDM_GPU_LAYERS=str(a.gpu_layers),
                           CDM_PREFILL_COUNT=str(max(1, a.prefill_count)),
                           CDM_INCREMENTAL='1' if incremental else '0')
                env.pop('CDM_UNROLLED_ROUTER', None)
                if a.unrolled_router:
                    env['CDM_UNROLLED_ROUTER'] = '1'
                proc = subprocess.run(
                    [str(a.runner.resolve()), str(Path(filename).resolve()),
                     str(Path(tmp) / 'logits.f32'), *map(str, ids)],
                    capture_output=True, text=True, env=env, check=True, timeout=600)
            if a.gpu_layers > 0 and not re.search(r'offloaded [1-9]\d*/\d+ layers to GPU', proc.stderr):
                raise ValueError('GPU requested but native logs do not confirm layer offload')
            match = re.search(r'^CDM_TIMES:\s*(.+)$', proc.stderr, re.M)
            if not match:
                raise ValueError('Runner did not return timing samples')
            times = [float(v) for v in match.group(1).split()]
            if len(times) != a.repeats:
                raise ValueError('Runner timing count differs from requested repeats')
            row = {'gguf': filename, 'mode': 'incremental_kv' if incremental else 'prefill',
                   'seconds': times,
                   'median_seconds': float(np.median(times)),
                   'tokens_per_second': len(ids) / float(np.median(times))}
            row['runner_environment'] = {k: env[k] for k in ['CDM_THREADS', 'CDM_REPEATS', 'CDM_GPU_LAYERS', 'CDM_PREFILL_COUNT', 'CDM_INCREMENTAL', 'CDM_UNROLLED_ROUTER', 'CUDA_VISIBLE_DEVICES', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_FORCE_MMQ', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'] if k in env}
            row['device_log'] = [line for line in proc.stderr.splitlines()
                                 if re.search(r'CUDA|offload|graph splits|buffer size', line)]
            decode = re.search(r'^CDM_DECODE_TIMES:\s*(.+)$', proc.stderr, re.M)
            if incremental and a.prefill_count:
                if not decode:
                    raise ValueError('Runner lacks separate post-prefill decode timings')
                decode_times = [float(v) for v in decode.group(1).split()]
                if len(decode_times) != a.repeats:
                    raise ValueError('Decode timing count mismatch')
                row['decode_seconds'] = decode_times
                row['decode_tokens'] = len(ids) - a.prefill_count
                row['decode_tokens_per_second'] = row['decode_tokens'] / float(np.median(decode_times))
            if not incremental:
                row['prefill_speedup_vs_pytorch'] = (
                    row['tokens_per_second'] / result['pytorch_prefill_tokens_per_second'])
            result['native'].append(row)
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
