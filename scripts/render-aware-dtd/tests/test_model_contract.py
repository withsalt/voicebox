from __future__ import annotations

import numpy as np
import torch

from aec_dataset import _shift
from dtd_model import DtdModelConfig, RenderAwareDtd


def test_expanded_model_matches_production_budget():
    model = RenderAwareDtd(DtdModelConfig())
    complexity = model.complexity()

    assert model.config.frequency_bands == 48
    assert model.config.projection_size == 192
    assert model.config.hidden_size == 192
    assert model.config.gru_layers == 2
    assert complexity.trainable_parameters == 510_337
    assert complexity.total_macs_per_second == 31_492_800


def test_streaming_shapes_match_contract():
    model = RenderAwareDtd().eval()
    microphone = torch.zeros(1, 640)
    render = torch.zeros(1, 640)
    state = model.initial_state(1)

    with torch.inference_mode():
        logits, next_state = model(microphone, render, state)

    assert logits.shape == (1, 1)
    assert state.shape == (2, 1, 192)
    assert next_state.shape == state.shape


def test_256_by_3_model_complexity_and_streaming_state():
    model = RenderAwareDtd(
        DtdModelConfig(
            projection_size=256,
            hidden_size=256,
            gru_layers=3,
        )
    ).eval()
    complexity = model.complexity()
    state = model.initial_state(1)

    with torch.inference_mode():
        logits, next_state = model(
            torch.zeros(1, 640),
            torch.zeros(1, 640),
            state,
        )

    assert complexity.trainable_parameters == 1_280_001
    assert complexity.total_macs_per_second == 69_843_200
    assert logits.shape == (1, 1)
    assert state.shape == (3, 1, 256)
    assert next_state.shape == state.shape


def test_shift_larger_than_signal_returns_silence():
    samples = np.arange(8, dtype=np.float32)

    assert np.count_nonzero(_shift(samples, 8)) == 0
    assert np.count_nonzero(_shift(samples, -9)) == 0
