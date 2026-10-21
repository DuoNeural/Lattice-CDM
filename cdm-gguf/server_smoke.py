#!/usr/bin/env python3
"""Exercise standard llama-server HTTP completion with the native CDM engine."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(ROOT / 'models/sub17m-q8_0-text.gguf'))
    parser.add_argument('--port', type=int, default=18099)
    parser.add_argument('--output', default=str(ROOT / 'research/http-q8-smoke.json'))
    parser.add_argument('--server', default=str(ROOT / 'llama.cpp/build/bin/llama-server'))
    parser.add_argument('--gpu-layers', type=int, default=0)
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'
    with tempfile.TemporaryFile(mode='w+') as log:
        process = subprocess.Popen([
            args.server, '-m', args.model, '-lv', '4',
            '-ngl', str(args.gpu_layers), '-t', '4', '-c', '512', '--host', '127.0.0.1',
            '--port', str(args.port)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError('llama-server exited before becoming ready')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1) as response:
                        health = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('llama-server startup timeout')
                    time.sleep(.2)
            payload = {'prompt': '<|im_start|>user\nWrite a short greeting.<|im_end|>\n<|im_start|>assistant\n',
                       'n_predict': 8, 'temperature': 0}
            request = urllib.request.Request(base + '/completion',
                                             data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=30) as response:
                completion = json.load(response)
            if completion.get('error') or not completion.get('content'):
                raise RuntimeError('Completion returned an error or empty content')
            log.flush()
            log.seek(0)
            device_log = log.read()
            if args.gpu_layers > 0 and 'layers to GPU' not in device_log:
                raise RuntimeError('GPU requested but server logs do not confirm offload')
            result = {'model': args.model, 'server': args.server,
                      'gpu_layers_requested': args.gpu_layers,
                      'backend_log': device_log,
                      'health': health, 'request': payload,
                      'completion': completion}
            Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'health': health, 'content': completion['content'],
                              'tokens_predicted': completion.get('tokens_predicted')}))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.seek(0)
            if process.returncode not in (0, -15):
                print(log.read()[-3000:])


if __name__ == '__main__':
    main()
