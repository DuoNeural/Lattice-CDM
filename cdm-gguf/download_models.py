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
    'lattice4_nexus': ('DuoNeural/Lattice-4-Nexus-253M-Poincare-Instruct', 'f9777f69c64badf98c0877bd5afe8c090bdaeab7', ['cdm_lm_distill_v3.py', 'pytorch_model.pt']),
    'lattice4_josephson': ('DuoNeural/Lattice-4-Josephson-253M-Poincare-Instruct', 'fcfe1a6936702f2bc562319ccfe7b465c398fab6', ['cdm_lm_distill_v3.py', 'pytorch_model.pt']),
    'lattice3_synapse': ('DuoNeural/Lattice-3-Synapse-192M-Poincare-Instruct', '27e0e8efd9320c638f3f4309983b76a4d2a731b4', ['cdm_lm_distill_v3.py', 'pytorch_model.pt']),
    'lattice2_base': ('DuoNeural/Lattice-2-192M-Poincare-Base', 'aed4eef2530795b111e777fc1432e18ca216a034', ['cdm_lm_distill_v2.py', 'pytorch_model.pt']),
    'sub18m_v3': ('DuoNeural/Sub-Lattice-v3-18M-Poincare-Hybrid', '47cccf7e4c00dae411edb956c2faf3111097fafe', ['cdm_lm_distill_sublattice_v3.py', 'pytorch_model.pt']),
}


def main():
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=[*RELEASES, 'new', 'all'], default='all')
    parser.add_argument('--directory', type=Path,
                        default=Path(__file__).resolve().parent / 'models')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    new_models = ['lattice4_nexus', 'lattice4_josephson', 'lattice3_synapse', 'lattice2_base', 'sub18m_v3']
    selected = RELEASES if args.model == 'all' else ({k: RELEASES[k] for k in new_models} if args.model == 'new' else {args.model: RELEASES[args.model]})
    manifest = {}
    for name, (repo, revision, files) in selected.items():
        manifest[name] = {'repo': repo, 'revision': revision, 'files': {}}
        for filename in files:
            cached = Path(hf_hub_download(repo, filename, revision=revision))
            source_dir = Path(__file__).resolve().parent / 'research' / 'model_sources'
            source_dir.mkdir(parents=True, exist_ok=True)
            if name in ('sub17m', 'lattice107m'):
                output = args.directory / filename
            else:
                output = (args.directory / f'{name}-pytorch_model.pt'
                          if filename.endswith('.pt') else source_dir / f'{name}-{filename}')
            if not output.exists():
                if filename.endswith('.pt'):
                    output.hardlink_to(cached)
                else:
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
