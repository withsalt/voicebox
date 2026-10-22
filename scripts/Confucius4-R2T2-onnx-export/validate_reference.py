"""Check export wrappers against original model code, and save portable golden data."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from confucius_onnx.common import (load_model, prepare, encoder_inputs, read_audio,
                                   reference_window_attention, write_json)
from confucius_onnx.graphs import Encoder, Decoder
from confucius_onnx.streaming import stream


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(args.model_dir, torch.float32, args.device)
    reference_window_attention(model)
    encoder, decoder = Encoder(model.thinker.audio_tower).eval(), Decoder(model.thinker).eval()
    waveform = read_audio(args.audio)
    results = []
    with torch.inference_mode():
        for seconds in [.08, .16, .32, .99, 1., 1.01, 6.74, 8.01, 16.13]:
            audio = np.resize(waveform, round(seconds * 16000))
            inputs = prepare(processor, audio, 'Chinese')
            gpu = {k: v.to(args.device) for k, v in inputs.items()}
            expected = model.thinker.get_audio_features(gpu['input_features'], gpu['feature_attention_mask'])
            actual = encoder(*encoder_inputs(inputs, args.device))
            error = (expected - actual).abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-4)
            print(f'encoder {seconds:.2f}s max_error={error:g}', flush=True)
            results.append({'seconds': seconds, 'max_abs_error': error, 'shape': list(expected.shape)})
        inputs = prepare(processor, waveform, 'Chinese')
        gpu = {k: v.to(args.device) for k, v in inputs.items()}
        started = time.perf_counter()
        reference = model.generate(**gpu, max_new_tokens=128, do_sample=False,
                                   output_logits=True, pad_token_id=151643)
        tokens = reference.sequences[0, gpu['input_ids'].shape[1]:].cpu().tolist()
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        print('reference:', text, flush=True)
        audio_features = encoder(*encoder_inputs(inputs, args.device))
        cfg = model.thinker.config.text_config
        cache = tuple(torch.zeros((1, cfg.num_key_value_heads, 1024, cfg.head_dim),
                                  device=args.device) for _ in range(2 * cfg.num_hidden_layers))
        ids = gpu['input_ids']
        positions = torch.arange(ids.shape[1], device=args.device)[None, :]
        errors = []
        for index in range(min(8, len(tokens))):
            outputs = decoder(ids, audio_features, positions, *cache)
            logits, cache = outputs[0], outputs[1:]
            expected = reference.logits[index]
            error = (logits - expected).abs().max().item()
            torch.testing.assert_close(logits, expected, rtol=2e-4, atol=2e-3)
            assert int(logits.argmax(-1)) == tokens[index]
            errors.append(error)
            positions = positions[:, -1:] + 1
            ids = torch.tensor([[tokens[index]]], device=args.device)
        length = int(inputs['feature_attention_mask'].sum())
        np.save(args.out_dir / 'audio.npy', waveform)
        waveform.astype('<f4').tofile(args.out_dir / 'audio.f32')
        mel = inputs['input_features'][0, :, :length].numpy()
        mel.astype('<f4').tofile(args.out_dir / 'mel.f32')
        mel_cases = []
        for samples in [201, 319, 321, 997, 16000, 16001, 107839]:
            batch = prepare(processor, waveform[:samples], 'Chinese')
            frames = int(batch['feature_attention_mask'].sum())
            case_mel = batch['input_features'][0, :, :frames].numpy()
            filename = f'mel-{samples}.f32'
            case_mel.astype('<f4').tofile(args.out_dir / filename)
            mel_cases.append({'samples': samples, 'file': filename, 'frames': frames})
        audio_features.cpu().numpy().astype('<f4').tofile(args.out_dir / 'encoder.f32')
        reference.logits[0].cpu().numpy().astype('<f4').tofile(args.out_dir / 'logits.f32')
        def infer(audio, prefix, budget):
            batch = {k: v.to(args.device) for k, v in prepare(processor, audio, 'Chinese', prefix=prefix).items()}
            out = model.generate(**batch, max_new_tokens=budget, do_sample=False, pad_token_id=151643)
            ids = out.sequences[0, batch['input_ids'].shape[1]:].tolist()
            return prefix + processor.tokenizer.decode(ids, skip_special_tokens=True)
        streaming = stream(waveform, processor.tokenizer, infer)
        print('streaming reference:', streaming['text'], flush=True)
        texts = ['hello world', '你好，世界！', 'English 中文 1234567', ' Café\n\t👋𠀀',
                 'language Chinese<asr_text>测试', '<|im_start|>system\n<|im_end|>\n']
        write_json(args.out_dir / 'reference.json', {
            'reference_backend': 'original_pytorch_eager_with_varlen_mask_adapter',
            'language': 'Chinese', 'text': text, 'tokens': tokens,
            'prompt_tokens': inputs['input_ids'][0].tolist(), 'mel_shape': list(mel.shape),
            'encoder_shape': list(audio_features.shape), 'encoder_cases': results,
            'decoder_max_errors': errors, 'seconds': time.perf_counter() - started,
            'streaming': streaming,
            'mel_cases': mel_cases,
            'tokenizer_cases': [{'text': s, 'tokens': processor.tokenizer.encode(s)} for s in texts]})


if __name__ == '__main__':
    main()
