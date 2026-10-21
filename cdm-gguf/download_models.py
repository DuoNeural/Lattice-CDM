#!/usr/bin/env python3
"""Download the two reviewed release revisions, preserving source provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RELEASES = {
    'sub17m': ('DuoNeural/Sub-Lattice-17M-Poincare-Instruct',
               'd601229f6103cedd3f8cecca6a378f250cc2ef5c',
               ['cdm_lm_distill_sublattice_v1.py',
                'cdm_sublattice_v1_poincare_final_step0100000.pt']),
    'lattice107m': ('DuoNeural/Lattice-1-107M-Poincare-Preview',
                   '0a846b092b0544f5abfcaa3891e1e88ceacc5e44',
                   ['cdm_lm_distill_v1.py', 'pytorch_model.pt']),
}


def main():
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[*RELEASES, 'all'], default='all')
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).resolve().parent / 'models')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    selected = RELEASES if args.model == 'all' else {args.model: RELEASES[args.model]}
    manifest = {}
    for name, (repo, revision, files) in selected.items():
        manifest[name] = {'repo': repo, 'revision': revision, 'files': {}}
        for filename in files:
            cached = Path(hf_hub_download(repo, filename, revision=revision))
            output = args.directory / filename
            if not output.exists():
                shutil.copyfile(cached, output)
            with cached.open('rb') as handle:
                expected = hashlib.file_digest(handle, 'sha256').hexdigest()
            with output.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'Existing {output} differs from pinned release')
            manifest[name]['files'][filename] = {'sha256': actual,
                                                'bytes': output.stat().st_size}
    result = args.directory / ('sources-' + args.model + '.json')
    result.write_text(json.dumps(manifest, indent=2) + '\n')
    print(result)


if __name__ == '__main__':
    main()
