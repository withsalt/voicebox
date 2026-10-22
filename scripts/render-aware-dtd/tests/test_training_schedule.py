from __future__ import annotations

from argparse import Namespace

import pytest

from train_dtd import learning_rate_for_epoch, remaining_training_epochs


def schedule_args(**overrides):
    values = {
        "epochs": 100,
        "learning_rate": 3e-4,
        "lr_schedule": "cosine",
        "warmup_epochs": 5,
        "min_learning_rate": 1e-5,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cosine_schedule_warms_up_and_reaches_minimum():
    args = schedule_args()

    assert learning_rate_for_epoch(args, 0) == pytest.approx(6e-5)
    assert learning_rate_for_epoch(args, 4) == pytest.approx(3e-4)
    assert learning_rate_for_epoch(args, 5) == pytest.approx(3e-4)
    assert learning_rate_for_epoch(args, 99) == pytest.approx(1e-5)


def test_constant_schedule_ignores_warmup():
    args = schedule_args(lr_schedule="constant")

    assert learning_rate_for_epoch(args, 0) == pytest.approx(3e-4)
    assert learning_rate_for_epoch(args, 99) == pytest.approx(3e-4)


def test_resume_runs_no_epoch_after_persisted_early_stop_boundary():
    epochs = remaining_training_epochs(
        start_epoch=42,
        total_epochs=100,
        patience=15,
        epochs_without_improvement=15,
    )

    assert list(epochs) == []
