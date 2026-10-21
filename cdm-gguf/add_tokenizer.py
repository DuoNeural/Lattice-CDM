#!/usr/bin/env python3
"""Attach a pinned teacher tokenizer to an ID-only CDM GGUF.

Uses the pinned llama.cpp tokenizer converter, including pretokenizer fingerprint
detection and special-token handling. Does not load teacher model weights or run
remote Python code. Writes a separate output file; never modifies the input.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'llama.cpp'), str(ROOT / 'llama.cpp/gguf-py')]


def attach(source, destination, repo, revision):
    import gguf
    from huggingface_hub import HfApi, snapshot_download
    from conversion.base import TextModel
    from gguf.scripts.gguf_new_metadata import copy_with_new_metadata

    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Choose a new output path distinct from the input')
    resolved = HfApi().model_info(repo, revision=revision).sha
    folder = Path(snapshot_download(
        repo, revision=resolved,
        allow_patterns=['tokenizer.json', 'tokenizer_config.json',
                        'special_tokens_map.json', 'added_tokens.json',
                        'vocab.json', 'merges.txt', 'config.json',
                        'chat_template.jinja'],
    ))
    reader = gguf.GGUFReader(source)
    if reader.get_field('general.architecture').contents() != 'cdm':
        raise ValueError('Expected a CDM GGUF')
    vocab = int(reader.get_field('cdm.vocab_size').contents())
    writer = gguf.GGUFWriter(destination, 'cdm')

    class TokenizerAdapter:
        get_vocab_base = TextModel.get_vocab_base
        get_vocab_base_pre = TextModel.get_vocab_base_pre
        does_token_look_special = TextModel.does_token_look_special

    adapter = TokenizerAdapter()
    adapter.dir_model = folder
    adapter.hparams = {'vocab_size': vocab}
    adapter.gguf_writer = writer
    TextModel._set_vocab_gpt2(adapter)
    special = gguf.SpecialVocab(folder, load_merges=True, n_vocab=vocab)
    if not special.merges:
        raise ValueError('Tokenizer has no BPE merges; refusing unusable output')
    # SpecialVocab includes the template from tokenizer_config where supplied.
    template_file = folder / 'chat_template.jinja'
    if special.chat_template is None and template_file.exists():
        writer.add_chat_template(template_file.read_text())
    writer.add_string('cdm.tokenizer.source', repo)
    writer.add_string('cdm.tokenizer.revision', resolved)
    remove = [key for key in reader.fields if key.startswith('tokenizer.')
              or key in ('cdm.tokenizer.source', 'cdm.tokenizer.revision')]
    copy_with_new_metadata(reader, writer, {}, remove)
    result = {'output': str(destination), 'tokenizer': repo,
              'revision': resolved, 'vocab_size': vocab}
    destination.with_suffix(destination.suffix + '.tokenizer.json').write_text(
        json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', default='main',
                        help='Resolved to an immutable commit and recorded')
    args = parser.parse_args()
    print(json.dumps(attach(args.input, args.output, args.repo, args.revision), indent=2))


if __name__ == '__main__':
    main()
