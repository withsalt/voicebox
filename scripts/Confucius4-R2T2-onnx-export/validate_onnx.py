import argparse
import json
from pathlib import Path
import numpy as np
from confucius_onnx.common import prepare, write_json
from confucius_onnx.runtime import Runtime
from confucius_onnx.streaming import stream


def main():
    p = argparse.ArgumentParser(description='Compare ONNX with independent original-PyTorch golden data.')
    p.add_argument('--package-dir', type=Path, required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--provider', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    reference = json.loads((args.reference_dir / 'reference.json').read_text('utf-8'))
    waveform = np.load(args.reference_dir / 'audio.npy')
    runtime = Runtime(args.package_dir, args.provider)
    inputs = prepare(runtime.processor, waveform, 'Chinese')
    expected = np.fromfile(args.reference_dir / 'encoder.f32', dtype='<f4').reshape(reference['encoder_shape'])
    actual = runtime.encode(inputs).astype(np.float32)
    encoder_error = float(np.max(np.abs(expected - actual)))
    result = runtime.transcribe(waveform, return_debug=True)
    print(f'Offline {runtime.metadata["dtype"]}/{args.provider}: {result["seconds"]:.3f}s', flush=True)
    logits = result.pop('first_logits').astype(np.float32)
    expected_logits = np.fromfile(args.reference_dir / 'logits.f32', dtype='<f4').reshape(logits.shape)
    result.update({'encoder_max_error': encoder_error,
                   'logits_max_error': float(np.max(np.abs(logits - expected_logits))),
                   'reference_text': reference['text'], 'exact_text': result['text'] == reference['text'],
                   'dtype': runtime.metadata['dtype'], 'provider': args.provider})
    result['streaming'] = stream(waveform, runtime.processor.tokenizer,
        lambda audio, prefix, budget: runtime.transcribe(audio, prefix=prefix, max_tokens=budget)['text'])
    if runtime.metadata['dtype'] == 'float32':
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=1e-3)
        np.testing.assert_allclose(logits, expected_logits, atol=5e-3, rtol=1e-3)
        assert result['exact_text'], result
        assert result['streaming']['text'] == reference['streaming']['text']
    elif runtime.metadata['dtype'] == 'float16':
        assert np.isfinite(actual).all() and np.isfinite(logits).all()
        assert result['exact_text'], result
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
