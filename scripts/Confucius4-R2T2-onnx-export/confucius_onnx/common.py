import json
from pathlib import Path

import numpy as np
import torch
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor


def load_model(model_dir, dtype, device):
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(model_dir), dtype=dtype, attn_implementation='eager',
        device_map=device, local_files_only=True).eval()
    processor = Qwen3ASRProcessor.from_pretrained(str(model_dir), local_files_only=True)
    return model, processor


def prompt(processor, language=None, context='', prefix=''):
    text = processor.apply_chat_template([
        {'role': 'system', 'content': context},
        {'role': 'user', 'content': [{'type': 'audio', 'audio': ''}]}],
        tokenize=False, add_generation_prompt=True)
    if language:
        text += f'language {language}<asr_text>'
    return text + prefix


def prepare(processor, audio, language=None, context='', prefix=''):
    return processor(text=[prompt(processor, language, context, prefix)],
                     audio=[np.asarray(audio, dtype=np.float32)],
                     return_tensors='pt', padding=True)


def encoder_inputs(inputs, device='cpu', dtype=torch.float32):
    length = int(inputs['feature_attention_mask'][0].sum())
    mel = inputs['input_features'][0, :, :length].to(device=device, dtype=dtype)
    chunks = (length + 99) // 100
    width = min(100, length)
    padded = torch.nn.functional.pad(mel, (0, chunks * width - length))
    padded = padded.reshape(128, chunks, width).permute(1, 0, 2).unsqueeze(1).contiguous()
    lengths = torch.full((chunks,), 100, dtype=torch.int64, device=device)
    lengths[-1] = length - (chunks - 1) * 100
    return padded, lengths


def reference_window_attention(model):
    """Restore varlen block masking for the upstream eager reference backend.

    qwen-asr 0.0.6 passes cu_seqlens to eager attention, which ignores it.
    vLLM/FlashAttention use those boundaries. This adapter only constructs the
    mask; convolution, layer norms, attention, projections, and decode still run
    the original model implementation, independently of our export wrappers.
    """
    def add_mask(module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        lengths = kwargs['cu_seqlens']
        groups = torch.repeat_interleave(torch.arange(len(lengths) - 1, device=hidden.device),
                                        lengths[1:] - lengths[:-1])
        blocked = groups[:, None] != groups[None, :]
        mask = torch.zeros(blocked.shape, dtype=hidden.dtype, device=hidden.device)
        kwargs['attention_mask'] = mask.masked_fill(blocked, torch.finfo(hidden.dtype).min)[None, None]
        return args, kwargs
    return [layer.self_attn.register_forward_pre_hook(add_mask, with_kwargs=True)
            for layer in model.thinker.audio_tower.layers]


def read_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    import math
    audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError('Audio contains non-finite samples')
    return audio


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
