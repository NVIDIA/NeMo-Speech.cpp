# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the CTC timestamp decoder and its tokenizer to GGUF (for the speech-LLM's encoder file,
conversion/salm.py).

The source is a CTC timestamp .nemo adapter, or a safetensors file holding NeMo's CTC timestamp
artifact: a full checkpoint or an extract with the ctc_timestamp.* tensors and metadata, or a
standalone artifact (decoder.*, tokenizer.model_proto). The frozen training encoder is deliberately
omitted. Default weights are F32.

An upsampling decoder (upsampling_factor U > 1: 40 ms CTC frames from 80 ms encoder states) also
gets its ConvTranspose1d(D, D, kernel_size=U, stride=U) as ctc.upsample, stored as a linear layer
of D inputs and U x D outputs: out[j * D + o] = b[o] + sum_i W[i, o, j] x[i] (PyTorch weight
(in, out, U)), so the U CTC frames of an encoder frame are consecutive D-rows of one product.
"""
import base64
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llama.cpp/gguf-py'))
from gguf import GGUFWriter

ARTIFACT_FORMAT = 'nemo_ctc_timestamp_artifact_v1'
EXPECTED = dict(
    use_transformer=True,
    pre_block_norm=True,
    self_attention_model='rope',
    rope_base=10000.0,
    xscaling=False,
    attn_mode='full',
    residual=True,
)


def read_nemo(source):
    """Decoder config, decoder tensors and SentencePiece model of a .nemo adapter."""
    with tarfile.open(source) as archive:

        def read(suffix):
            matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {suffix}')
            return archive.extractfile(matches[0])

        cfg = yaml.safe_load(read('model_config.yaml'))['decoder']
        spm = read('tokenizer.model').read()
        state = torch.load(read('model_weights.ckpt'), map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
    weights = {k.removeprefix('decoder.'): v for k, v in state.items() if k.startswith('decoder.')}
    return cfg, weights, spm


def read_safetensors(source):
    """The same from a safetensors CTC timestamp artifact (bundled under ctc_timestamp. or
    standalone), reading only its tensors."""
    from safetensors import safe_open

    with safe_open(str(source), framework='pt', device='cpu') as f:
        meta = f.metadata() or {}
        if meta.get('ctc_timestamp_format') == ARTIFACT_FORMAT:
            prefix, keys = 'ctc_timestamp.', 'ctc_timestamp_'
        elif meta.get('format') == ARTIFACT_FORMAT:
            prefix, keys = '', ''
        else:
            raise ValueError(f'{source}: no {ARTIFACT_FORMAT} metadata')
        cfg = json.loads(meta[keys + 'decoder_config'])
        tokenizer = json.loads(meta[keys + 'tokenizer_config'])
        decoder = prefix + 'decoder.'
        weights = {
            k.removeprefix(decoder): f.get_tensor(k) for k in f.keys() if k.startswith(decoder)
        }
        spm = f.get_tensor(prefix + 'tokenizer.model_proto').numpy().tobytes()
    # NeMo's SentencePieceTokenizer encodes plain SentencePiece ids only with these options (legacy
    # mode splits on registered special tokens; kept extra whitespace takes another path).
    if tokenizer.get('legacy', False) or not tokenizer.get('ignore_extra_whitespaces', True):
        raise ValueError(f'Unsupported CTC tokenizer options {tokenizer}')
    return cfg, weights, spm


def convert(source, output, model_name=None):
    source = Path(source)
    if tarfile.is_tarfile(source):
        cfg, weights, spm = read_nemo(source)
    else:
        cfg, weights, spm = read_safetensors(source)
    for key, value in EXPECTED.items():
        if cfg.get(key) != value:
            raise ValueError(f'Unsupported decoder {key}={cfg.get(key)}; expected {value}')
    factor = int(cfg.get('upsampling_factor', 1))
    if factor < 1 or cfg.get('upsampling_mode', 'conv_transpose') != 'conv_transpose':
        raise ValueError(f'Unsupported upsampling {factor} {cfg.get("upsampling_mode")}')
    if factor > 1 and int(weights.pop('upsample_factor')) != factor:
        raise ValueError('upsample_factor buffer does not match the decoder config')
    writer = GGUFWriter(str(output), 'ctc_timestamp')
    d = cfg['feat_in']
    for key, value in dict(
        d_model=d,
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        ff_hidden=int(d * cfg['ff_expansion']),
        rotary_dims=int(d // cfg['n_heads'] * cfg['rotary_fraction']),
        vocab_size=cfg['num_classes'] + 1,
        upsampling_factor=factor,
    ).items():
        writer.add_uint32('ctc.' + key, value)
    writer.add_bool('ctc.qk_norm', cfg['qk_norm'])
    writer.add_bool('ctc.qkv_bias', cfg['qkv_bias'])
    writer.add_float32('ctc.residual_scale', float(weights.pop('residual_scale')))
    writer.add_string('ctc.tokenizer.spm_model', base64.b64encode(spm).decode())
    if model_name:  # reported as the alignment model (else a name derived from the vocabulary)
        writer.add_string('ctc.model_name', model_name)
    # Store only the adapter; no precision reduction in this initial port.
    for name, value in weights.items():
        if name == 'decoder_layers.0.weight':
            name, value = 'classifier.weight', value.squeeze(-1)
        elif name == 'decoder_layers.0.bias':
            name = 'classifier.bias'
        elif name == 'upsample.weight' and factor > 1:
            # (in, out, U) -> rows j * D + o, columns i
            value = value.permute(2, 1, 0).reshape(factor * d, d)
        elif name == 'upsample.bias' and factor > 1:
            value = value.repeat(factor)
        elif not name.startswith('transformer.'):
            raise ValueError(f'Unrecognized decoder tensor {name}')
        writer.add_tensor('ctc.' + name, value.float().contiguous().numpy().astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(
        f'{output}: {len(weights)} tensors, {cfg["num_classes"]} tokenizer pieces, '
        f'upsampling {factor}, F32'
    )
