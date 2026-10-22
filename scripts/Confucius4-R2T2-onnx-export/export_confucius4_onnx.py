"""Export a self-contained, versioned Confucius4 package for VoiceBox / ORT."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from confucius_onnx.common import load_model, prepare, encoder_inputs, write_json
from confucius_onnx.graphs import Encoder, Decoder


def tensors_in_graph(graph):
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from tensors_in_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from tensors_in_graph(child)


def consolidate(source, destination):
    """Stream external weights; avoid loading a second 8 GB model into memory."""
    model = onnx.load(str(source), load_external_data=False)
    data_path = destination.with_suffix('.onnx.data')
    with data_path.open('wb') as target:
        for tensor in tensors_in_graph(model.graph):
            # Shape inference reads small integer constants before resolving
            # external Constant attributes; retain these inline in the graph.
            if tensor.data_location != onnx.TensorProto.EXTERNAL and tensor.ByteSize() < 4096:
                continue
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                small_info = {x.key: x.value for x in tensor.external_data}
                if int(small_info.get('length', '4096')) < 4096:
                    original = (source.parent / small_info['location']).resolve()
                    if not original.is_relative_to(source.parent.resolve()):
                        raise ValueError('External tensor escapes staging directory')
                    with original.open('rb') as stream:
                        stream.seek(int(small_info.get('offset', 0)))
                        data = stream.read(int(small_info['length']))
                    if len(data) != int(small_info['length']):
                        raise IOError('Truncated small external tensor')
                    tensor.ClearField('external_data')
                    tensor.data_location = onnx.TensorProto.DEFAULT
                    tensor.raw_data = data
                    continue
            offset = target.tell()
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                info = {x.key: x.value for x in tensor.external_data}
                original = (source.parent / info['location']).resolve()
                if not original.is_relative_to(source.parent.resolve()):
                    raise ValueError(f'External tensor escapes staging directory: {original}')
                with original.open('rb') as stream:
                    stream.seek(int(info.get('offset', 0)))
                    remaining = int(info.get('length', original.stat().st_size - stream.tell()))
                    while remaining:
                        block = stream.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise IOError(f'Truncated external tensor: {tensor.name}')
                        target.write(block)
                        remaining -= len(block)
            else:
                target.write(numpy_helper.to_array(tensor).tobytes())
            size = target.tell() - offset
            for field in ('raw_data', 'float_data', 'int32_data', 'int64_data',
                          'double_data', 'uint64_data', 'external_data'):
                tensor.ClearField(field)
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for key, value in [('location', data_path.name), ('offset', str(offset)), ('length', str(size))]:
                entry = tensor.external_data.add()
                entry.key, entry.value = key, value
    onnx.save(model, str(destination))
    onnx.checker.check_model(str(destination))


def export_graph(module, args, destination, names, outputs, dynamic):
    with tempfile.TemporaryDirectory(prefix='.graph-', dir=destination.parent) as temp:
        source = Path(temp) / destination.name
        with torch.inference_mode():
            torch.onnx.export(module, args, str(source), input_names=names, output_names=outputs,
                              dynamic_axes=dynamic, opset_version=20, dynamo=False,
                              external_data=True, do_constant_folding=True)
        consolidate(source, destination)
    print(f'Exported {destination.name}', flush=True)


def quantize(path):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
    with tempfile.TemporaryDirectory(prefix='.quant-', dir=path.parent) as temp:
        result = Path(temp) / path.name
        quantizer = MatMulNBitsQuantizer(str(path), algo_config=DefaultWeightOnlyQuantConfig(
            block_size=32, is_symmetric=True, accuracy_level=1, bits=8,
            op_types_to_quantize=('MatMul',)))
        quantizer.process()
        quantizer.model.save_model_to_file(str(result), use_external_data_format=True)
        del quantizer
        gc.collect()
        consolidate(result, path)
    graph = onnx.load(str(path), load_external_data=False)
    count = sum(n.op_type == 'MatMulNBits' for n in graph.graph.node)
    if count == 0:
        raise RuntimeError(f'No weights were quantized in {path.name}')
    return count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--dtype', choices=['fp32', 'fp16', 'int8'], default='fp16')
    parser.add_argument('--device', default='cpu', help='PyTorch export device; cpu avoids GPU export peaks.')
    parser.add_argument('--cache-length', type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(8)
    src, out = args.model_dir.resolve(), args.out_dir.resolve()
    if not (src / 'model.safetensors').is_file():
        raise FileNotFoundError(f'model.safetensors not found: {src}')
    if out.is_relative_to(src) or src.is_relative_to(out):
        raise ValueError('Model and output directories must not contain each other')
    if out.exists():
        raise FileExistsError(f'Output already exists; select a new directory: {out}')
    if args.cache_length < 128:
        raise ValueError('--cache-length must be at least 128')
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.export-', dir=out.parent))
    started = time.perf_counter()
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    try:
        model, processor = load_model(src, dtype, args.device)
        config = model.thinker.config
        if (config.audio_config.n_window != 50 or config.audio_config.num_mel_bins != 128 or
                config.audio_config.output_dim != 2048 or config.text_config.hidden_size != 2048 or
                config.audio_config.n_window_infer <= 0 or config.audio_config.n_window_infer % 100):
            raise ValueError('This package schema requires the Confucius4 128-bin / 2048-dimension architecture')
        if args.cache_length > config.text_config.max_position_embeddings:
            raise ValueError('--cache-length exceeds model position capacity')
        rng = np.random.default_rng(42)
        inputs = prepare(processor, rng.normal(0, .01, 19760).astype(np.float32), 'Chinese')
        padded, lengths = encoder_inputs(inputs, args.device, dtype)
        encoder = Encoder(model.thinker.audio_tower).eval()
        with torch.inference_mode():
            features = encoder(padded, lengths)
        export_graph(encoder, (padded, lengths), stage / 'encoder.onnx',
                     ['padded_feature', 'chunk_lengths'], ['audio_features'],
                     {'padded_feature': {0: 'chunks', 3: 'chunk_frames'}, 'chunk_lengths': {0: 'chunks'},
                      'audio_features': {0: 'audio_tokens'}})
        tc = config.text_config
        ids = inputs['input_ids'].to(args.device)
        cache = tuple(torch.zeros(1, tc.num_key_value_heads, args.cache_length, tc.head_dim,
                                 dtype=dtype, device=args.device) for _ in range(2 * tc.num_hidden_layers))
        past = [f'past_{kind}_{i:02d}' for i in range(tc.num_hidden_layers) for kind in ['key', 'value']]
        present = [s.replace('past_', 'present_') for s in past]
        positions = torch.arange(ids.shape[1], device=args.device)[None, :]
        export_graph(Decoder(model.thinker).eval(), (ids, features, positions, *cache),
                     stage / 'decoder.onnx', ['input_ids', 'audio_features', 'cache_position', *past],
                     ['logits', *present], {'input_ids': {1: 'query_tokens'},
                                          'audio_features': {0: 'audio_tokens'},
                                          'cache_position': {1: 'query_tokens'}})
        processor.save_pretrained(stage)
        # Embed exact mel filters so C# does not depend on a second implementation.
        preprocessor = json.loads((stage / 'preprocessor_config.json').read_text('utf-8'))
        preprocessor['mel_filters'] = processor.feature_extractor.mel_filters.T.tolist()
        write_json(stage / 'preprocessor_config.json', preprocessor)
        config_copy = json.loads((src / 'config.json').read_text('utf-8'))
        write_json(stage / 'config.json', config_copy)
        for name in ['generation_config.json', 'tokenizer.json', 'added_tokens.json', 'special_tokens_map.json']:
            if (src / name).is_file():
                shutil.copy2(src / name, stage / name)
        license_root = Path(__file__).parent / 'licenses'
        for name in ['LICENSE', 'MODEL_LICENSE', 'MODEL_LICENSE_zh']:
            license_file = src / name if (src / name).is_file() else license_root / name
            if not license_file.is_file():
                raise FileNotFoundError(f'Required license file missing: {name}')
            shutil.copy2(license_file, stage / name)
        metadata = {
            'schema_version': 1, 'model_family': 'confucius4-r2t2',
            'dtype': {'fp32': 'float32', 'fp16': 'float16', 'int8': 'int8'}[args.dtype],
            'opset': 20, 'sample_rate': 16000, 'encoder_model': 'encoder.onnx',
            'prefill_model': 'decoder.onnx', 'decode_model': 'decoder.onnx',
            'decode_cache_mode': 'full', 'shared_decoder_session': True,
            'static_cache_len': args.cache_length, 'num_layers': tc.num_hidden_layers,
            'num_key_value_heads': tc.num_key_value_heads, 'head_dim': tc.head_dim,
            'audio_output_dim': config.audio_config.output_dim, 'n_window': 50,
            'dynamic_chunk_frames': True,
            'n_window_infer': config.audio_config.n_window_infer,
            'audio_attention': 'bidirectional_windows', 'audio_token_id': config.audio_token_id,
            'audio_token': processor.tokenizer.audio_token, 'eos_token_ids': [151643, 151645],
            'languages': config_copy['support_languages'],
            'streaming': {'chunk_ms': 160, 'lookahead_ms': 160, 'rollback_tokens': 1,
                          'max_window_seconds': 16, 'decode_tokens': 4},
            'versions': {p: importlib.metadata.version(p) for p in
                         ['torch', 'transformers', 'qwen-asr', 'onnx', 'onnxruntime-gpu']},
        }
        del model, encoder, cache, features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.dtype == 'int8':
            metadata['quantization'] = {'method': 'weight_only_blockwise', 'weight_bits': 8,
                                        'activation_dtype': 'float32', 'block_size': 32,
                                        'accuracy_level': 1, 'operator_domain': 'com.microsoft',
                                        'operators': {name: quantize(stage / name)
                                                      for name in ['encoder.onnx', 'decoder.onnx']}}
        # The checker alone does not detect stale external offsets inside Constant
        # attributes. Load each final graph with ORT before publishing the directory.
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 8
        session_options.log_severity_level = 3
        for name in ['encoder.onnx', 'decoder.onnx']:
            session = ort.InferenceSession(str(stage / name), session_options, providers=['CPUExecutionProvider'])
            del session
        metadata['validation'] = {'onnx_checker': True, 'ort_cpu_session_load': True}
        metadata['files'] = {}
        for path in sorted(stage.iterdir()):
            if path.is_file():
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                metadata['files'][path.name] = {'bytes': path.stat().st_size, 'sha256': digest}
        metadata['export_seconds'] = time.perf_counter() - started
        write_json(stage / 'metadata.json', metadata)
        stage.rename(out)
        print(f'Package: {out} ({metadata["export_seconds"]:.1f}s)', flush=True)
    except BaseException:
        print(f'Export failed. Incomplete staging files retained for diagnosis: {stage}', flush=True)
        raise


if __name__ == '__main__':
    main()
