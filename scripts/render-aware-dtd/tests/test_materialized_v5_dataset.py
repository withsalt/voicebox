from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from materialized_v5_dataset import MaterializedV5Dataset


def write_manifest(root, name: str, row: dict[str, object]) -> None:
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{name}.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )


def test_materialized_v5_reads_triplet_and_reset_metadata(tmp_path):
    sample_count = 16_000
    microphone = np.zeros(sample_count, dtype=np.float32)
    render = np.full(sample_count, 0.02, dtype=np.float32)
    target = np.zeros(sample_count, dtype=np.float32)
    target[8_000:12_000] = 0.1
    audio = tmp_path / "triplet.wav"
    sf.write(
        audio,
        np.column_stack((microphone + target, render, target)),
        16_000,
        subtype="PCM_16",
    )
    write_manifest(
        tmp_path,
        "materialized_train",
        {
            "audio": str(audio),
            "file_id": "triplet",
            "mode": "mixture",
            "scenario": "reset_cold_onset",
            "ser_db": -9.0,
            "reset_frame": 10,
            "gap_hops": 2,
            "cluster_id": "near:test:one",
        },
    )

    item = MaterializedV5Dataset(
        tmp_path,
        "materialized_train",
        sequence_seconds=1.0,
    )[0]

    assert item["microphone"].shape == (sample_count,)
    assert item["render"].shape == (sample_count,)
    assert item["target"].shape == (49,)
    assert item["has_reset"] is True
    assert item["reset_before"][10]
    assert item["gap_hops"] == 2
    assert item["source"] == "materialized_v5:reset_cold_onset"


def test_materialized_v5_rejects_wrong_channel_count(tmp_path):
    audio = tmp_path / "mono.wav"
    sf.write(audio, np.zeros(16_000, dtype=np.float32), 16_000)
    write_manifest(
        tmp_path,
        "materialized_train",
        {
            "audio": str(audio),
            "file_id": "mono",
            "mode": "far_only",
            "ser_db": -99.0,
        },
    )

    dataset = MaterializedV5Dataset(
        tmp_path,
        "materialized_train",
        sequence_seconds=1.0,
    )
    with pytest.raises(ValueError, match="3 channels"):
        dataset[0]
