from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from torch.utils.data import Dataset

import enriched_dataset
from enriched_dataset import (
    OnlineSyntheticDataset,
    SynthesisDistribution,
    V5StreamAugmentation,
    create_enriched_datasets,
    labels_for,
)


def write_manifest(root, name: str, row: dict[str, object]) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{name}.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )


def write_tone(path, sample_rate: int, seconds: float, channels: int = 1) -> None:
    sample_count = round(sample_rate * seconds)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2.0 * np.pi * 330.0 * time)
    samples = tone if channels == 1 else np.stack([tone, tone * 0.8], axis=1)
    sf.write(path, samples, sample_rate, subtype="FLOAT")


def test_labels_follow_propagated_nearend():
    nearend = np.zeros(16_000, dtype=np.float32)
    nearend[8_000:12_000] = 0.1

    labels = labels_for(nearend)

    assert np.count_nonzero(labels[:20]) == 0
    assert np.count_nonzero(labels[24:40]) > 0


def test_online_synthesis_accepts_mixed_source_rates(tmp_path):
    farend = tmp_path / "farend-48k.wav"
    nearend = tmp_path / "nearend-44k.wav"
    noise = tmp_path / "noise-16k.wav"
    rir = tmp_path / "rir-48k.wav"
    write_tone(farend, 48_000, 1.2, channels=2)
    write_tone(nearend, 44_100, 0.6)
    write_tone(noise, 16_000, 1.2)
    impulse = np.zeros(4_800, dtype=np.float32)
    impulse[0] = 1.0
    impulse[240] = 0.3
    sf.write(rir, impulse, 48_000, subtype="FLOAT")

    for split in ("train", "validation", "test"):
        write_manifest(
            tmp_path,
            f"farend_{split}",
            {"path": str(farend), "corpus": "far", "file_id": "far"},
        )
        write_manifest(
            tmp_path,
            f"nearend_{split}",
            {"path": str(nearend), "corpus": "near", "file_id": "near"},
        )
        write_manifest(
            tmp_path,
            f"noise_{split}",
            {"path": str(noise), "corpus": "noise", "file_id": "noise"},
        )
        write_manifest(
            tmp_path,
            f"rir_{split}",
            {"path": str(rir), "corpus": "rir", "file_id": "rir"},
        )

    dataset = OnlineSyntheticDataset(
        tmp_path,
        split="train",
        examples=2,
        sequence_seconds=1.0,
        seed=7,
        distribution=SynthesisDistribution(
            far_only_probability=0.0,
            near_only_probability=0.0,
            noise_probability=1.0,
        ),
        stream_augmentation=V5StreamAugmentation(reset_probability=1.0),
    )
    item = dataset[0]

    assert item["microphone"].shape == (16_000,)
    assert item["render"].shape == (16_000,)
    assert item["target"].shape == (49,)
    assert np.isfinite(item["microphone"].numpy()).all()
    assert np.isfinite(item["render"].numpy()).all()
    assert item["has_reset"] is True
    assert item["gap_hops"] > 0
    reset_before = item["reset_before"].numpy()
    valid_mask = item["valid_mask"].numpy()
    assert np.count_nonzero(reset_before) == 2
    reset_frame = int(np.flatnonzero(reset_before)[1])
    expected_invalid = np.zeros_like(valid_mask)
    expected_invalid[reset_frame - 1] = True
    expected_invalid[-int(item["gap_hops"]) :] = True
    assert np.array_equal(~valid_mask, expected_invalid)


def test_enriched_dataset_weights_are_validated_before_loading(tmp_path):
    with pytest.raises(ValueError, match="dataset_weights"):
        create_enriched_datasets(
            tmp_path,
            sequence_seconds=1.0,
            examples_per_epoch=2,
            validation_examples=2,
            seed=7,
            dataset_weights=(0.5, -0.1, 0.4, 0.2),
        )


@pytest.mark.parametrize("invalid_weight", [float("nan"), float("inf")])
def test_enriched_dataset_weights_must_be_finite(tmp_path, invalid_weight):
    with pytest.raises(ValueError, match="finite"):
        create_enriched_datasets(
            tmp_path,
            sequence_seconds=1.0,
            examples_per_epoch=2,
            validation_examples=2,
            seed=7,
            dataset_weights=(0.4, 0.2, invalid_weight, 0.2),
        )


@pytest.mark.parametrize("probability", [-0.01, 0.900001, float("nan")])
def test_wenet_far_only_probability_is_validated_before_loading(
    tmp_path,
    probability,
):
    with pytest.raises(ValueError, match="far_only|single-talk"):
        create_enriched_datasets(
            tmp_path,
            sequence_seconds=1.0,
            examples_per_epoch=2,
            validation_examples=2,
            seed=7,
            wenet_far_only_probability=probability,
        )


def test_wenet_far_only_probability_only_changes_online_training(
    tmp_path,
    monkeypatch,
):
    class StubDataset(Dataset):
        def __init__(
            self,
            *args,
            distribution=None,
            stream_augmentation=None,
            **kwargs,
        ):
            self.distribution = distribution
            self.stream_augmentation = stream_augmentation

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise AssertionError("stub samples must not be read")

    monkeypatch.setattr(enriched_dataset, "ManifestStemDataset", StubDataset)
    monkeypatch.setattr(enriched_dataset, "OnlineSyntheticDataset", StubDataset)
    monkeypatch.setattr(enriched_dataset, "RealSingleTalkDataset", StubDataset)

    train, validation, description = create_enriched_datasets(
        tmp_path,
        sequence_seconds=1.0,
        examples_per_epoch=2,
        validation_examples=2,
        seed=7,
        wenet_far_only_probability=0.63,
        v5_reset_probability=0.1,
        v5_hard_far_only_probability=0.25,
        v5_validation_reset_probability=0.2,
        v5_validation_hard_far_only_probability=0.5,
    )

    assert train.datasets[2].distribution.far_only_probability == 0.63
    assert validation.datasets[3].distribution.far_only_probability == 0.50
    assert description["online_train_distribution"]["far_only_probability"] == 0.63
    assert train.datasets[2].stream_augmentation.reset_probability == 0.1
    assert train.datasets[2].stream_augmentation.hard_far_only_probability == 0.25
    assert validation.datasets[3].stream_augmentation.reset_probability == 0.2
    assert validation.datasets[3].stream_augmentation.hard_far_only_probability == 0.5
