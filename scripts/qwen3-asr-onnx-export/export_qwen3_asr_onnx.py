#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnx
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
DEFAULT_EOS_TOKEN_IDS = [151645, 151643]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a full single-audio ONNX inference pipeline for Qwen3-ASR. "
            "Export phase uses PyTorch, but runtime can stay on ONNX Runtime + tokenizer/feature extractor."
        )
    )
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for ONNX files and metadata.",
    )
    parser.add_argument("--device", default="cuda:0", help="Export device.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "int8"],
        default="float16",
        help="Export dtype. `int8` exports a float32 graph and applies dynamic int8 weight quantization.",
    )
    parser.add_argument(
        "--int8-weight-type",
        choices=["qint8", "quint8"],
        default="qint8",
        help="Weight tensor type used for int8 dynamic quantization (only effective when --dtype=int8).",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Dummy audio length used for tracing.",
    )
    parser.add_argument(
        "--static-cache-len",
        type=int,
        default=1664,
        help="Fixed KV cache length for exported decoder graph.",
    )
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing ONNX files."
    )
    parser.add_argument(
        "--cache-mode",
        choices=["delta", "full"],
        default="delta",
        help=(
            "KV cache 输出模式。"
            "delta（默认，CPU 路径下最快）：导出 prefill.onnx + decode.onnx 双图，decode 仅输出每步增量"
            "（D2H 量 ~115KB/步，CPU 路径下最快）。"
            "full：导出单张 prefill.onnx 兼任 decode（cache 输入/输出 shape 一致），可让 CUDA"
            "+ IOBinding 把 KV cache 完全驻留 GPU，RTF 约降低 百分之45；CPU 路径会变慢。"
        ),
    )
    parser.add_argument(
        "--worklog", default="WORKLOG.md", help="Optional markdown log."
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "int8": torch.float32}[
        name
    ]


def quantize_onnx_int8(onnx_path: Path, weight_type_name: str) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    weight_type = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}[
        weight_type_name
    ]
    parent = onnx_path.parent
    final_data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    fp32_dir = parent / f"_fp32_{onnx_path.stem}"
    if fp32_dir.exists():
        shutil.rmtree(fp32_dir)
    fp32_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = fp32_dir / onnx_path.name
    fp32_data_path = fp32_path.with_suffix(fp32_path.suffix + ".data")
    quant_dir = parent / f"_int8_{onnx_path.stem}"
    if quant_dir.exists():
        shutil.rmtree(quant_dir)
    quant_dir.mkdir(parents=True, exist_ok=True)
    quant_path = quant_dir / onnx_path.name

    shutil.move(str(onnx_path), str(fp32_path))
    if final_data_path.exists():
        shutil.move(str(final_data_path), str(fp32_data_path))

    try:
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(quant_path),
            weight_type=weight_type,
            use_external_data_format=True,
        )
        model = onnx.load(str(quant_path), load_external_data=True)
        if final_data_path.exists():
            final_data_path.unlink()
        # 同 export_graph: 避开 onnx 1.21.0 的 all_tensors_to_one_file 合并写入 bug。
        _write_external_data(model, final_data_path)
        onnx.save(model, str(onnx_path))
    finally:
        if fp32_dir.exists():
            shutil.rmtree(fp32_dir, ignore_errors=True)
        if quant_dir.exists():
            shutil.rmtree(quant_dir, ignore_errors=True)


def feature_output_lengths(lengths: torch.Tensor) -> torch.Tensor:
    leave = lengths % 100
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (lengths // 100) * 13


def build_prompt(processor: AutoProcessor) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def flatten_legacy_cache(
    cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for key_states, value_states in cache:
        flat.append(key_states)
        flat.append(value_states)
    return flat


class FixedSizeCacheLayer:
    def __init__(self, max_cache_len: int):
        self.max_cache_len = max_cache_len
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.last_key_update: torch.Tensor | None = None
        self.last_value_update: torch.Tensor | None = None

    def allocate_like(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        self.keys = torch.zeros(
            (
                key_states.shape[0],
                key_states.shape[1],
                self.max_cache_len,
                key_states.shape[3],
            ),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.zeros(
            (
                value_states.shape[0],
                value_states.shape[1],
                self.max_cache_len,
                value_states.shape[3],
            ),
            dtype=value_states.dtype,
            device=value_states.device,
        )

    def load(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.keys = key_states
        self.values = value_states
        self.last_key_update = None
        self.last_value_update = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, torch.Tensor] | None,
    ):
        if self.keys is None or self.values is None:
            self.allocate_like(key_states, value_states)
        self.last_key_update = key_states
        self.last_value_update = value_states
        cache_position = (
            cache_kwargs["cache_position"] if cache_kwargs is not None else None
        )
        if cache_position is None:
            cache_position = torch.arange(
                key_states.shape[-2], device=key_states.device
            ).view(1, -1)
        scatter_index = cache_position.view(
            key_states.shape[0], 1, key_states.shape[2], 1
        ).expand_as(key_states)
        self.keys.scatter_(2, scatter_index, key_states)
        self.values.scatter_(2, scatter_index, value_states)
        visible_len = torch.max(cache_position) + 1
        return self.keys[:, :, :visible_len, :], self.values[:, :, :visible_len, :]


class FixedSizeCache:
    def __init__(self, num_layers: int, max_cache_len: int):
        self.layers = [
            FixedSizeCacheLayer(max_cache_len=max_cache_len) for _ in range(num_layers)
        ]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, torch.Tensor] | None = None,
    ):
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


class DeltaCacheLayer:
    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.last_key_update: torch.Tensor | None = None
        self.last_value_update: torch.Tensor | None = None

    def load(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.keys = key_states
        self.values = value_states
        self.last_key_update = None
        self.last_value_update = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, torch.Tensor] | None,
    ):
        if self.keys is None or self.values is None:
            raise RuntimeError("Delta cache layer must be loaded before decode.")
        self.last_key_update = key_states
        self.last_value_update = value_states
        cache_position = (
            cache_kwargs["cache_position"] if cache_kwargs is not None else None
        )
        if cache_position is None:
            raise ValueError("Delta cache update requires cache_position.")
        visible_len = torch.max(cache_position) + 1
        scatter_index = cache_position.view(
            key_states.shape[0], 1, key_states.shape[2], 1
        ).expand_as(key_states)
        visible_keys = self.keys[:, :, :visible_len, :].scatter(
            2, scatter_index, key_states
        )
        visible_values = self.values[:, :, :visible_len, :].scatter(
            2, scatter_index, value_states
        )
        return visible_keys, visible_values


class DeltaOnlyCache:
    def __init__(self, num_layers: int):
        self.layers = [DeltaCacheLayer() for _ in range(num_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, torch.Tensor] | None = None,
    ):
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


def flatten_static_cache(cache: FixedSizeCache) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for layer in cache.layers:
        flat.append(layer.keys)
        flat.append(layer.values)
    return flat


def flatten_cache_updates(
    cache: FixedSizeCache, cache_position: torch.Tensor
) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for layer in cache.layers:
        if layer.keys is None or layer.values is None:
            raise RuntimeError("Cache tensors are unavailable.")
        gather_index = cache_position.view(cache_position.shape[0], 1, 1, 1).expand(
            layer.keys.shape[0],
            layer.keys.shape[1],
            1,
            layer.keys.shape[3],
        )
        flat.append(torch.gather(layer.keys, 2, gather_index).clone())
        flat.append(torch.gather(layer.values, 2, gather_index).clone())
    return flat


def flatten_delta_updates(cache: DeltaOnlyCache) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for layer in cache.layers:
        if layer.last_key_update is None or layer.last_value_update is None:
            raise RuntimeError("Delta cache layer has no updates.")
        flat.append(layer.last_key_update.clone())
        flat.append(layer.last_value_update.clone())
    return flat


def _write_external_data(model: onnx.ModelProto, data_path: Path) -> None:
    """把 model 中所有内联了 raw_data 的 initializer 顺序写到 data_path 这一个文件,
       并把每个 initializer 改写为指向该文件的 external_data 引用。

       这是 onnx.save_model(save_as_external_data=True, all_tensors_to_one_file=True)
       的等价替代 — 后者在 onnx 1.21.0 上有 bug, 会在合并写入时破坏个别 scalar。
       本实现按 initializer 顺序拼接 raw_data, 偏移连续, 与 ORT 的外部数据加载约定一致。
    """
    data_path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with open(data_path, "wb") as df:
        for init in model.graph.initializer:
            raw = init.raw_data
            if not raw:
                # 已经是外部数据或无数据 (例如字符串张量) — 保持原样。
                continue
            df.write(raw)
            length = len(raw)
            # 改写引用
            init.ClearField("raw_data")
            init.data_location = onnx.TensorProto.EXTERNAL
            del init.external_data[:]
            for k, v in (
                ("location", data_path.name),
                ("offset", str(offset)),
                ("length", str(length)),
            ):
                ed = init.external_data.add()
                ed.key = k
                ed.value = v
            offset += length


def export_graph(
    module: nn.Module,
    args_tuple: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    existing_files = {
        path.name for path in output_path.parent.iterdir() if path.is_file()
    }
    torch.onnx.export(
        module,
        args=args_tuple,
        f=str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=False,
        external_data=True,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    data_path = output_path.with_suffix(output_path.suffix + ".data")
    model = onnx.load(str(output_path), load_external_data=True)

    # 把权重外部化到单一 .data 文件。
    # 不能用 onnx.save_model(..., all_tensors_to_one_file=True, size_threshold=0) 或
    # onnx.external_data_helper.convert_model_to_external_data(..., all_tensors_to_one_file=True),
    # 它们在 onnx 1.21.0 上有 bug — 合并写入时会破坏个别 scalar
    # (实测: audio_tower.layers.14.self_attn.v_proj.bias[895] 会被改成 -8.5)。
    # 手工拼接 raw_data 并重设 external_data 引用绕开这条出错路径,
    # 与 save_model 的预期产物逐字节等价但不触发 bug。
    _write_external_data(model, data_path)
    onnx.save(model, str(output_path))
    current_files = [path for path in output_path.parent.iterdir() if path.is_file()]
    keep_names = existing_files | {output_path.name, data_path.name}
    for path in current_files:
        if path.name not in keep_names:
            path.unlink()


class ConvEmbedCore(nn.Module):
    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower

    def forward(self, padded_feature: torch.Tensor):
        x = torch.nn.functional.gelu(self.audio_tower.conv2d1(padded_feature))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d2(x))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d3(x))
        batch_size, channels, freq_bins, time_steps = x.shape
        x = self.audio_tower.conv_out(
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size, time_steps, channels * freq_bins)
        )
        pos = (
            self.audio_tower.positional_embedding.positional_embedding[: x.shape[1], :]
            .unsqueeze(0)
            .to(x.dtype)
        )
        return x + pos


class AudioStackCore(nn.Module):
    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower
        sample_attn = audio_tower.layers[0].self_attn
        self.num_heads = sample_attn.num_heads
        self.head_dim = sample_attn.head_dim
        self.scaling = sample_attn.scaling

    def self_attention(
        self, attn_mod: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        query = (
            attn_mod.q_proj(hidden_states)
            .view(seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        key = (
            attn_mod.k_proj(hidden_states)
            .view(seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        value = (
            attn_mod.v_proj(hidden_states)
            .view(seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_output = torch.matmul(attn_weights, value)
        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(seq_len, self.num_heads * self.head_dim)
            .contiguous()
        )
        return attn_mod.out_proj(attn_output)

    def forward(self, hidden_states: torch.Tensor):
        x = hidden_states
        for layer in self.audio_tower.layers:
            residual = x
            x = layer.self_attn_layer_norm(x)
            x = self.self_attention(layer.self_attn, x)
            x = residual + x

            residual = x
            x = layer.final_layer_norm(x)
            x = layer.fc1(x)
            x = layer.activation_fn(x)
            x = layer.fc2(x)
            x = residual + x

            if x.dtype == torch.float16:
                clamp_value = torch.finfo(x.dtype).max - 1000
                x = torch.clamp(x, min=-clamp_value, max=clamp_value)

        x = self.audio_tower.ln_post(x)
        x = self.audio_tower.proj1(x)
        x = self.audio_tower.act(x)
        x = self.audio_tower.proj2(x)
        return x


class EncoderCore(nn.Module):
    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.conv_embed_core = ConvEmbedCore(audio_tower)
        self.audio_stack_core = AudioStackCore(audio_tower)

    def forward(self, padded_feature: torch.Tensor, chunk_lengths: torch.Tensor):
        padded_embed = self.conv_embed_core(padded_feature)
        chunk_feature_lengths_after = feature_output_lengths(chunk_lengths)
        mask_after = torch.arange(
            padded_embed.shape[1], device=padded_embed.device
        ).unsqueeze(0) < chunk_feature_lengths_after.unsqueeze(1)
        hidden_states = padded_embed[mask_after]
        return self.audio_stack_core(hidden_states)


class DecoderCore(nn.Module):
    def __init__(
        self,
        thinker: nn.Module,
        audio_token_id: int,
        num_layers: int,
        max_cache_len: int,
    ):
        super().__init__()
        self.thinker = thinker
        self.text_model = thinker.model
        self.audio_token_id = audio_token_id
        self.num_layers = num_layers
        self.max_cache_len = max_cache_len

    def build_static_cache(
        self, past_key_values: Sequence[torch.Tensor]
    ) -> FixedSizeCache:
        cache = FixedSizeCache(
            num_layers=self.num_layers, max_cache_len=self.max_cache_len
        )
        for layer_idx, layer in enumerate(cache.layers):
            key_states = past_key_values[layer_idx * 2]
            value_states = past_key_values[layer_idx * 2 + 1]
            layer.load(key_states, value_states)
        return cache

    def build_inputs_embeds(
        self, input_ids: torch.Tensor, audio_features: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
        audio_mask = (
            (input_ids == self.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        return inputs_embeds.masked_scatter(
            audio_mask, audio_features.to(inputs_embeds.dtype)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        cache_position: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        cache = self.build_static_cache(past_key_values)
        inputs_embeds = self.build_inputs_embeds(input_ids, audio_features)
        position_ids = cache_position.unsqueeze(0).expand(3, -1, -1)
        text_position_ids = position_ids[0]
        visible_len = torch.max(cache_position) + 1

        key_positions = torch.arange(visible_len, device=input_ids.device)
        query_positions = cache_position
        blocked = key_positions.view(1, 1, visible_len) > query_positions.unsqueeze(-1)
        causal_mask = torch.zeros(
            (input_ids.shape[0], 1, input_ids.shape[1], visible_len),
            dtype=inputs_embeds.dtype,
            device=input_ids.device,
        )
        min_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask = causal_mask.masked_fill(blocked.unsqueeze(1), min_value)

        hidden_states = inputs_embeds
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
        for layer in self.text_model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.text_model.norm(hidden_states)
        last_hidden_state = hidden_states[:, -1:, :]
        logits = self.thinker.lm_head(last_hidden_state)
        return (logits[:, 0, :], *flatten_static_cache(cache))


class DecodeCore(nn.Module):
    def __init__(self, thinker: nn.Module, num_layers: int, max_cache_len: int):
        super().__init__()
        self.thinker = thinker
        self.text_model = thinker.model
        self.num_layers = num_layers
        self.max_cache_len = max_cache_len

    def build_delta_cache(
        self, past_key_values: Sequence[torch.Tensor]
    ) -> DeltaOnlyCache:
        cache = DeltaOnlyCache(num_layers=self.num_layers)
        for layer_idx, layer in enumerate(cache.layers):
            key_states = past_key_values[layer_idx * 2]
            value_states = past_key_values[layer_idx * 2 + 1]
            layer.load(key_states, value_states)
        return cache

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        cache = self.build_delta_cache(past_key_values)
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
        position_ids = cache_position.unsqueeze(0).expand(3, -1, -1)
        text_position_ids = position_ids[0]
        visible_len = torch.max(cache_position) + 1

        key_positions = torch.arange(visible_len, device=input_ids.device)
        query_positions = cache_position
        blocked = key_positions.view(1, 1, visible_len) > query_positions.unsqueeze(-1)
        causal_mask = torch.zeros(
            (input_ids.shape[0], 1, input_ids.shape[1], visible_len),
            dtype=inputs_embeds.dtype,
            device=input_ids.device,
        )
        min_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask = causal_mask.masked_fill(blocked.unsqueeze(1), min_value)

        hidden_states = inputs_embeds
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
        for layer in self.text_model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.text_model.norm(hidden_states)
        last_hidden_state = hidden_states[:, -1:, :]
        logits = self.thinker.lm_head(last_hidden_state)
        return (logits[:, 0, :], *flatten_delta_updates(cache))


def ensure_overwrite_allowed(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing files: {joined}. Use --overwrite to replace them."
        )


def append_worklog(
    worklog_path: Path, model_name: str, output_dir: Path, dtype_name: str
) -> None:
    lines = [
        "",
        f"### 节点 6：ONNX 导出完成（{model_name}）",
        f"- 完成时间：`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "- 导出范围：单音频全链路纯 ONNX 运行时所需的 3 张图：`encoder`、`prefill`、`decode`。",
        "- 运行时依赖目标：`onnxruntime + transformers(tokenizer/feature_extractor) + numpy + soundfile/librosa`，不依赖 PyTorch。",
        "- 说明：音频编码链路已合并为单一 `encoder.onnx`，内部仍使用同权重的 ONNX 友好实现。",
        f"- 导出 dtype：`{dtype_name}`",
        f"- 导出目录：`{output_dir}`",
        "- decoder 图说明：导出为 `prefill + decode` 双图，`decode` 仅保留逐 token 必需输入，以降低多并发逐 token 开销。",
    ]
    with worklog_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def cleanup_legacy_exports(output_dir: Path, cache_mode: str) -> None:
    names = [
        "conv_embed.onnx",
        "conv_embed.onnx.data",
        "audio_stack.onnx",
        "audio_stack.onnx.data",
        "decoder.onnx",
        "decoder.onnx.data",
    ]
    # full 模式下 prefill.onnx 兼任 decode，残留的 decode.onnx 会让 C# runtime 误以为
    # 还是 delta 模式（SharedDecoder=false），所以这里主动清理。
    if cache_mode == "full":
        names += ["decode.onnx", "decode.onnx.data"]
    for name in names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def bundle_runtime_assets(
    model_path: Path, output_dir: Path, processor: AutoProcessor
) -> None:
    processor.save_pretrained(str(output_dir))
    chat_template_path = model_path / "chat_template.json"
    if chat_template_path.exists():
        shutil.copy2(chat_template_path, output_dir / "chat_template.json")
    generation_config_path = model_path / "generation_config.json"
    if generation_config_path.exists():
        shutil.copy2(generation_config_path, output_dir / "generation_config.json")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        output_dir / "encoder.onnx",
        output_dir / "prefill.onnx",
        output_dir / "decode.onnx",
        output_dir / "metadata.json",
    ]
    ensure_overwrite_allowed(paths, overwrite=args.overwrite)

    export_dtype = resolve_dtype(args.dtype)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=export_dtype,
        device_map=args.device,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(str(model_path), fix_mistral_regex=True)
    thinker = model.thinker.eval()
    audio_tower = thinker.audio_tower.eval()
    text_config = thinker.config.text_config
    audio_config = thinker.config.audio_config

    prompt = build_prompt(processor)
    dummy_audio = np.zeros(int(args.sample_seconds * 16000), dtype=np.float32)
    inputs = processor(
        text=[prompt], audio=[dummy_audio], return_tensors="pt", padding=True
    )
    input_ids = inputs["input_ids"].to(args.device)
    input_features = inputs["input_features"].to(
        device=args.device, dtype=export_dtype
    )[0]
    feature_attention_mask = inputs["feature_attention_mask"].to(args.device)[0]
    feature_len = int(feature_attention_mask.sum().item())

    window = audio_config.n_window * 2
    chunk_num = (feature_len + window - 1) // window
    padded_total = chunk_num * window
    feature_slice = input_features[:, :feature_len]
    if padded_total > feature_len:
        feature_slice = torch.nn.functional.pad(
            feature_slice, (0, padded_total - feature_len)
        )
    padded_feature = (
        feature_slice.transpose(0, 1)
        .reshape(chunk_num, window, feature_slice.shape[0])
        .transpose(1, 2)
        .unsqueeze(1)
        .contiguous()
    )
    chunk_lengths = torch.full(
        (chunk_num,), window, device=input_features.device, dtype=torch.long
    )
    chunk_lengths[-1] = feature_len - window * (chunk_num - 1)

    encoder_core = EncoderCore(audio_tower).eval()
    with torch.no_grad():
        audio_features = encoder_core(padded_feature, chunk_lengths)

    export_graph(
        encoder_core,
        args_tuple=(padded_feature, chunk_lengths),
        output_path=output_dir / "encoder.onnx",
        input_names=["padded_feature", "chunk_lengths"],
        output_names=["audio_features"],
        dynamic_axes={
            "padded_feature": {0: "num_chunks", 3: "chunk_frames"},
            "chunk_lengths": {0: "num_chunks"},
            "audio_features": {0: "audio_seq"},
        },
        opset=args.opset,
    )

    sample_attn = thinker.model.layers[0].self_attn
    num_layers = len(thinker.model.layers)
    num_key_value_heads = int(sample_attn.config.num_key_value_heads)
    head_dim = int(sample_attn.head_dim)
    present_names = [
        f"present_key_{i:02d}" if p == 0 else f"present_value_{i:02d}"
        for i in range(num_layers)
        for p in range(2)
    ]
    past_names = [
        f"past_key_{i:02d}" if p == 0 else f"past_value_{i:02d}"
        for i in range(num_layers)
        for p in range(2)
    ]
    empty_cache = tuple(
        torch.zeros(
            (1, num_key_value_heads, args.static_cache_len, head_dim),
            device=input_ids.device,
            dtype=export_dtype,
        )
        for _ in range(num_layers * 2)
    )
    cache_position = torch.arange(
        input_ids.shape[1], device=input_ids.device, dtype=torch.long
    ).view(1, -1)
    decoder_core = DecoderCore(
        thinker,
        audio_token_id=thinker.config.audio_token_id,
        num_layers=num_layers,
        max_cache_len=args.static_cache_len,
    ).eval()
    export_graph(
        decoder_core,
        args_tuple=(input_ids, audio_features, cache_position, *empty_cache),
        output_path=output_dir / "prefill.onnx",
        input_names=["input_ids", "audio_features", "cache_position", *past_names],
        output_names=["logits", *present_names],
        dynamic_axes={
            "input_ids": {1: "query_seq"},
            "audio_features": {0: "audio_seq"},
            "cache_position": {1: "query_seq"},
            "logits": {0: "batch"},
        },
        opset=args.opset,
    )

    if args.cache_mode == "delta":
        # delta 模式：单独导出 decode.onnx，decode 步骤输出 [batch, heads, 1, head_dim]
        # 的增量 KV，C# runtime 走 Dictionary 路径在 CPU 端 scatter。
        decode_batch = 2
        decode_input_ids = input_ids[:, :1].repeat(decode_batch, 1).contiguous()
        decode_cache_position = torch.full(
            (decode_batch, 1),
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )
        decode_empty_cache = tuple(
            torch.zeros(
                (decode_batch, num_key_value_heads, args.static_cache_len, head_dim),
                device=input_ids.device,
                dtype=export_dtype,
            )
            for _ in range(num_layers * 2)
        )
        decode_core = DecodeCore(
            thinker,
            num_layers=num_layers,
            max_cache_len=args.static_cache_len,
        ).eval()
        export_graph(
            decode_core,
            args_tuple=(decode_input_ids, decode_cache_position, *decode_empty_cache),
            output_path=output_dir / "decode.onnx",
            input_names=["input_ids", "cache_position", *past_names],
            output_names=["logits", *present_names],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "cache_position": {0: "batch"},
                "logits": {0: "batch"},
                **{name: {0: "batch"} for name in past_names},
                **{name: {0: "batch"} for name in present_names},
            },
            opset=args.opset,
        )
    else:
        # full 模式：prefill.onnx 已经是 DecoderCore（输出 full KV cache），其 dynamic_axes
        # 允许 input_ids/cache_position 第 1 维变化，可同时承担 prefill（query_seq=prompt_len）
        # 与 decode（query_seq=1）。decode 时 input_ids 不含 audio_token → DecoderCore 内的
        # masked_scatter 不会读取 audio_features（传一个 shape-correct 的零占位即可），
        # C# runtime 已经在 _persistentAudioPlaceholder 里准备好了这个占位。
        pass

    if args.dtype == "int8":
        graphs_to_quantize = ["encoder.onnx", "prefill.onnx"]
        if args.cache_mode == "delta":
            graphs_to_quantize.append("decode.onnx")
        for graph_name in graphs_to_quantize:
            quantize_onnx_int8(
                output_dir / graph_name, weight_type_name=args.int8_weight_type
            )

    metadata = {
        "model_path": str(model_path),
        "model_name": model_path.name,
        "dtype": args.dtype,
        "device": args.device,
        "opset": args.opset,
        "eos_token_ids": DEFAULT_EOS_TOKEN_IDS,
        "sample_seconds": args.sample_seconds,
        "runtime_mode": "single_audio_full_onnx_no_torch",
        "audio_token_id": int(thinker.config.audio_token_id),
        "audio_token": processor.tokenizer.audio_token,
        "feature_size": int(audio_config.num_mel_bins),
        "n_window": int(audio_config.n_window),
        "n_window_infer": int(audio_config.n_window_infer),
        "audio_d_model": int(audio_config.d_model),
        "audio_output_dim": int(audio_config.output_dim),
        "text_hidden_size": int(text_config.hidden_size),
        "num_layers": num_layers,
        "static_cache_len": int(args.static_cache_len),
        "encoder_model": "encoder.onnx",
        "decoder_model": "prefill.onnx",
        "prefill_model": "prefill.onnx",
        # full 模式下 prefill.onnx 兼任 decode，C# 的 Qwen3AsrSessions 通过比较
        # prefill_model == decode_model 来判定 SharedDecoder=true，进而触发
        # IOBinding GPU-resident KV cache 路径（见 Qwen3AsrDecoder ctor 的
        # `useGpuCache && !_deltaCacheMode && _sessions.SharedDecoder` 三条件）。
        "decode_model": "prefill.onnx" if args.cache_mode == "full" else "decode.onnx",
        "shared_decoder_session": args.cache_mode == "full",
        "decode_batch_mode": "dynamic",
        "decode_cache_mode": args.cache_mode,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if args.dtype == "int8":
        metadata["int8_weight_type"] = args.int8_weight_type
        metadata["int8_quantization"] = "dynamic"
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    bundle_runtime_assets(
        model_path=model_path, output_dir=output_dir, processor=processor
    )
    cleanup_legacy_exports(output_dir, cache_mode=args.cache_mode)

    if args.worklog:
        append_worklog(
            Path(args.worklog).resolve(),
            model_name=model_path.name,
            output_dir=output_dir,
            dtype_name=args.dtype,
        )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
