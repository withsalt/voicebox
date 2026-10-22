"""Tensor-only ONNX boundaries. Stream scheduling belongs to the host runtime.

The acoustic attention mask preserves Qwen3-ASR's varlen / vLLM window semantics.
The decoder shares its weights between prefill and single-token decoding. Cache
updates are functional, so ORT need not alias an input and an output allocation.
"""
import torch
from torch import nn


def feature_lengths(lengths):
    tail = lengths % 100
    return (((tail - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1 + lengths // 100 * 13


class Encoder(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.tower = tower
        self.heads = tower.config.encoder_attention_heads
        self.dim = tower.config.d_model // self.heads
        self.window_tokens = 13 * (tower.config.n_window_infer // 100)

    def forward(self, padded_feature, chunk_lengths):
        t = self.tower
        x = torch.nn.functional.gelu(t.conv2d1(padded_feature))
        x = torch.nn.functional.gelu(t.conv2d2(x))
        x = torch.nn.functional.gelu(t.conv2d3(x))
        b, channels, freq, time = x.shape
        x = t.conv_out(x.permute(0, 3, 1, 2).reshape(b, time, channels * freq))
        x = x + t.positional_embedding.positional_embedding[:time].to(x.dtype)
        valid = torch.arange(time, device=x.device)[None, :] < feature_lengths(chunk_lengths)[:, None]
        x = x[valid]
        n = x.shape[0]
        groups = torch.arange(n, device=x.device) // self.window_tokens
        blocked = groups[:, None] != groups[None, :]
        for layer in t.layers:
            residual = x
            h = layer.self_attn_layer_norm(x)
            attn = layer.self_attn
            q = attn.q_proj(h).reshape(n, self.heads, self.dim).transpose(0, 1)
            k = attn.k_proj(h).reshape(n, self.heads, self.dim).transpose(0, 1)
            v = attn.v_proj(h).reshape(n, self.heads, self.dim).transpose(0, 1)
            scores = (q @ k.transpose(-1, -2)) * attn.scaling
            scores = scores.masked_fill(blocked[None], torch.finfo(scores.dtype).min)
            probs = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
            h = (probs @ v).transpose(0, 1).reshape(n, self.heads * self.dim)
            x = residual + attn.out_proj(h)
            residual = x
            h = layer.final_layer_norm(x)
            x = residual + layer.fc2(layer.activation_fn(layer.fc1(h)))
            if x.dtype == torch.float16:
                x = x.clamp(-64504, 64504)
        return t.proj2(t.act(t.proj1(t.ln_post(x))))


class TensorCache:
    def __init__(self, values):
        self.values = list(values)

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        positions = cache_kwargs['cache_position']
        index = positions.reshape(1, 1, -1, 1).expand_as(keys)
        self.values[2 * layer_idx] = self.values[2 * layer_idx].scatter(2, index, keys)
        self.values[2 * layer_idx + 1] = self.values[2 * layer_idx + 1].scatter(2, index, values)
        end = positions.max() + 1
        return (self.values[2 * layer_idx][:, :, :end, :],
                self.values[2 * layer_idx + 1][:, :, :end, :])


class Decoder(nn.Module):
    def __init__(self, thinker):
        super().__init__()
        self.thinker = thinker

    def forward(self, input_ids, audio_features, cache_position, *past):
        thinker = self.thinker
        model = thinker.model
        x = thinker.get_input_embeddings()(input_ids)
        mask = (input_ids == thinker.config.audio_token_id).unsqueeze(-1).expand_as(x)
        x = x.masked_scatter(mask, audio_features.to(x.dtype))
        cache = TensorCache(past)
        positions = cache_position.unsqueeze(0).expand(3, -1, -1)
        end = cache_position.max() + 1
        keys = torch.arange(end, device=x.device)
        blocked = keys.reshape(1, 1, -1) > cache_position.unsqueeze(-1)
        causal = torch.zeros((1, 1, input_ids.shape[1], end), device=x.device, dtype=x.dtype)
        causal = causal.masked_fill(blocked.unsqueeze(1), torch.finfo(x.dtype).min)
        rotary = model.rotary_emb(x, positions)
        for layer in model.layers:
            x = layer(x, attention_mask=causal, position_ids=positions[0],
                      past_key_values=cache, cache_position=cache_position,
                      position_embeddings=rotary)
        logits = thinker.lm_head(model.norm(x)[:, -1, :])
        return (logits, *cache.values)
