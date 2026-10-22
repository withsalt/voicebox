from __future__ import annotations

from pathlib import Path

import soundfile as sf
from torch.utils.data import ConcatDataset, Dataset

import enriched_dataset
from aec_dataset import SAMPLE_RATE
from enriched_dataset import (
    DatasetItem,
    WeightedMixedDataset,
    labels_for,
    load_jsonl,
    resolve_audio_path,
    sequence_sample_count,
    standard_item,
)


class MaterializedV5Dataset(Dataset[DatasetItem]):
    """Load precomputed microphone/render/target triplets from one WAV file."""

    def __init__(
        self,
        root: str | Path,
        manifest_name: str,
        *,
        sequence_seconds: float,
        limit: int = 0,
    ):
        self.root = Path(root)
        self.rows = load_jsonl(self.root / "manifests" / f"{manifest_name}.jsonl")
        if limit > 0:
            self.rows = self.rows[:limit]
        self.sequence_samples = sequence_sample_count(sequence_seconds)

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def __getitem__(self, index: int) -> DatasetItem:
        row = self.rows[index]
        path = resolve_audio_path(self.root, row["audio"])
        try:
            signals, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError) as error:
            raise ValueError(f"cannot read materialized v5 audio: {path}") from error
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"materialized v5 audio must be {SAMPLE_RATE} Hz: {path}")
        if signals.shape != (self.sequence_samples, 3):
            raise ValueError(
                "materialized v5 audio must contain exactly "
                f"{self.sequence_samples} frames and 3 channels: {path} "
                f"has {signals.shape}"
            )
        microphone = signals[:, 0]
        render = signals[:, 1]
        propagated_nearend = signals[:, 2]
        mode = str(row["mode"])
        if mode not in {"far_only", "near_only", "mixture"}:
            raise ValueError(f"invalid materialized v5 mode for {path}: {mode}")
        labels = labels_for(propagated_nearend)
        if mode == "far_only":
            labels.fill(0.0)
        return standard_item(
            microphone,
            render,
            labels,
            source="materialized_v5:" + str(row.get("scenario", mode)),
            mode=mode,
            file_id=str(row["file_id"]),
            ser_db=float(row["ser_db"]),
            farend_nonlinear=bool(row.get("farend_nonlinear", False)),
            farend_noisy=bool(row.get("farend_noisy", False)),
            nearend_noisy=bool(row.get("nearend_noisy", False)),
            reset_frame=int(row.get("reset_frame", 0)),
            gap_hops=int(row.get("gap_hops", 0)),
            cluster_id=str(row.get("cluster_id", row["file_id"])),
        )


def create_materialized_v5_datasets(
    root: str | Path,
    **kwargs: object,
) -> tuple[WeightedMixedDataset, Dataset[DatasetItem], dict[str, object]]:
    """Use v4 sources for three branches and replace only online synthesis."""

    train, validation, description = enriched_dataset.create_enriched_datasets(
        root, **kwargs
    )
    if not isinstance(train, WeightedMixedDataset):
        raise TypeError("expected the enriched training dataset to be weighted")
    if not isinstance(validation, ConcatDataset):
        raise TypeError("expected the enriched validation dataset to be concatenated")

    sequence_seconds = float(kwargs["sequence_seconds"])
    train_limit = int(kwargs.get("train_limit", 0))
    validation_limit = int(kwargs.get("validation_limit", 0))
    validation_examples = int(kwargs["validation_examples"])
    materialized_train = MaterializedV5Dataset(
        root,
        "materialized_train",
        sequence_seconds=sequence_seconds,
        limit=train_limit,
    )
    materialized_validation = MaterializedV5Dataset(
        root,
        "materialized_validation",
        sequence_seconds=sequence_seconds,
        limit=(validation_limit if validation_limit > 0 else validation_examples),
    )
    train.datasets[2] = materialized_train
    validation.datasets[3] = materialized_validation
    validation.cumulative_sizes = validation.cumsum(validation.datasets)

    description["schema_version"] = "3.0-materialized"
    description["online_train_source"] = "materialized_v5"
    description["online_validation_source"] = "materialized_v5"
    description["train_source_examples"]["wenet_online_synthetic_budget"] = len(  # type: ignore[index]
        materialized_train
    )
    description["validation_examples"] = len(validation)
    description["farend_corpus"] = (
        "materialized Wenet/BUT/AEC real mixtures; see dataset_summary.json"
    )
    return train, validation, description
