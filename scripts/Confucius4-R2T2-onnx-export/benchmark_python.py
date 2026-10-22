"""Original PyTorch or ORT dataset run. Reports preserve per-utterance outputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from confucius_onnx.common import load_model, prepare, reference_window_attention, write_json
from confucius_onnx.streaming import normalize, stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['torch', 'onnx'], required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--provider', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--mode', choices=['offline', 'stream'], default='offline')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = json.loads(args.manifest.read_text('utf-8'))
    if args.limit:
        cases = cases[:args.limit]
    if args.backend == 'torch':
        device = 'cuda:0' if args.provider == 'cuda' else 'cpu'
        model, processor = load_model(args.model_dir, torch.float32, device)
        reference_window_attention(model)
        def infer(audio, language, prefix='', budget=256):
            batch = {k: v.to(device) for k, v in prepare(processor, audio, language, prefix=prefix).items()}
            with torch.inference_mode():
                result = model.generate(**batch, max_new_tokens=budget, do_sample=False, pad_token_id=151643)
            return prefix + processor.tokenizer.decode(result.sequences[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    else:
        from confucius_onnx.runtime import Runtime
        runtime = Runtime(args.model_dir, args.provider)
        processor = runtime.processor
        def infer(audio, language, prefix='', budget=256):
            return runtime.transcribe(audio, language, prefix=prefix, max_tokens=budget)['text']
    infer(np.zeros(5120, np.float32), 'Chinese')
    results = []
    for i, case in enumerate(cases):
        audio = np.fromfile(case['audio'], dtype='<f4')
        start = time.perf_counter()
        if args.mode == 'stream':
            if len(audio) > 16 * 16000:
                raise ValueError('Use clips <=16s for direct single-window streaming parity')
            text = stream(audio, processor.tokenizer,
                          lambda chunk, prefix, budget: infer(chunk, case['language'], prefix, budget))['text']
        else:
            text = normalize(infer(audio, case['language'])).strip()
        elapsed = time.perf_counter() - start
        results.append({'id': case['id'], 'reference': case['text'], 'text': text,
                        'audio_seconds': len(audio)/16000, 'seconds': elapsed})
        print(f'{i+1}/{len(cases)} {case["id"]} {elapsed:.3f}s {text}', flush=True)
    write_json(args.out, {'backend': args.backend, 'provider': args.provider, 'mode': args.mode, 'cases': results,
                         'rtf': sum(x['seconds'] for x in results) / sum(x['audio_seconds'] for x in results)})


if __name__ == '__main__':
    main()
