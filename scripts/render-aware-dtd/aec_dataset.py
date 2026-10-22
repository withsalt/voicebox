from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_io import read_mono_resampled


SAMPLE_RATE = 16_000
FRAME_LENGTH = 640
HOP_LENGTH = 320
ABSOLUTE_FLOOR_RMS = 0.0005
RELATIVE_FLOOR = 0.03
MAXIMUM_GAP_FRAMES = 5


@dataclass(frozen=True)
class AecExample:
    file_id: int
    split: str
    nearend_scale: float
    ser_db: int
    farend_nonlinear: bool
    farend_noisy: bool
    nearend_noisy: bool


def load_examples(
    dataset_root: str | Path,
    split: str,
    validation_modulo: int = 20,
) -> list[AecExample]:
    synthetic_root = Path(dataset_root) / "datasets" / "synthetic"
    metadata_path = synthetic_root / "meta.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"AEC Challenge metadata not found: {metadata_path}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")
    if validation_modulo < 2:
        raise ValueError("validation_modulo must be >= 2")

    examples: list[AecExample] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            file_id = int(row["fileid"])
            source_split = row["split"].strip().lower()
            if split == "test":
                include = source_split == "test"
            elif split == "validation":
                include = source_split == "train" and file_id % validation_modulo == 0
            else:
                include = source_split == "train" and file_id % validation_modulo != 0
            if not include:
                continue
            examples.append(
                AecExample(
                    file_id=file_id,
                    split=source_split,
                    nearend_scale=float(row["nearend_scale"]),
                    ser_db=int(row["ser"]),
                    farend_nonlinear=bool(int(row["is_farend_nonlinear"])),
                    farend_noisy=bool(int(row["is_farend_noisy"])),
                    nearend_noisy=bool(int(row["is_nearend_noisy"])),
                )
            )
    return examples


def nearend_activity(samples: np.ndarray) -> tuple[np.ndarray, float]:
    frame_count = math.ceil(len(samples) / HOP_LENGTH)
    padded = np.pad(samples, (0, frame_count * HOP_LENGTH - len(samples)))
    frames = padded.reshape(frame_count, HOP_LENGTH)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    threshold = max(ABSOLUTE_FLOOR_RMS, float(rms.max(initial=0.0)) * RELATIVE_FLOOR)
    active = rms >= threshold

    active_indices = np.flatnonzero(active)
    for left, right in zip(active_indices[:-1], active_indices[1:]):
        gap = int(right - left - 1)
        if 0 < gap <= MAXIMUM_GAP_FRAMES:
            active[left + 1 : right] = True
    return active.astype(np.float32), threshold


def _shift(samples: np.ndarray, amount: int) -> np.ndarray:
    if amount == 0:
        return samples.copy()
    shifted = np.zeros_like(samples)
    if abs(amount) >= len(samples):
        return shifted
    if amount > 0:
        shifted[amount:] = samples[:-amount]
    else:
        shifted[:amount] = samples[-amount:]
    return shifted


def _echo_path_movement(echo: np.ndarray, rng: random.Random) -> np.ndarray:
    if len(echo) < HOP_LENGTH * 4:
        return echo
    change = rng.randrange(HOP_LENGTH, len(echo) - HOP_LENGTH)
    crossfade = min(HOP_LENGTH, change, len(echo) - change)
    shifted = _shift(echo, rng.randint(-160, 160))
    before_gain = 10.0 ** (rng.uniform(-3.0, 3.0) / 20.0)
    after_gain = 10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)
    moved = echo * before_gain
    after = shifted * after_gain
    moved[change:] = after[change:]
    if crossfade > 1:
        start = change - crossfade // 2
        end = start + crossfade
        blend = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
        moved[start:end] = (
            echo[start:end] * before_gain * (1.0 - blend)
            + after[start:end] * blend
        )
    return moved


class AecChallengeDtdDataset(Dataset[dict[str, torch.Tensor | int | str | bool]]):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        sequence_seconds: float | None = None,
        augment: bool = False,
        seed: int = 42,
        validation_modulo: int = 20,
        limit: int = 0,
    ):
        self.synthetic_root = Path(dataset_root) / "datasets" / "synthetic"
        self.examples = load_examples(dataset_root, split, validation_modulo)
        if limit > 0:
            self.examples = self.examples[:limit]
        self.split = split
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.sequence_samples = (
            None
            if sequence_seconds is None
            else max(
                FRAME_LENGTH,
                int(sequence_seconds * SAMPLE_RATE) // HOP_LENGTH * HOP_LENGTH,
            )
        )

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read(self, directory: str, name: str) -> np.ndarray:
        path = self.synthetic_root / directory / name
        return read_mono_resampled(path, target_sample_rate=SAMPLE_RATE)

    def __getitem__(
        self, index: int
    ) -> dict[str, torch.Tensor | int | str | bool]:
        example = self.examples[index]
        item_seed = (
            self.seed
            + self.epoch * 1_000_003
            + example.file_id * 9_176
            + index * 101
        )
        rng = random.Random(item_seed)
        numpy_rng = np.random.default_rng(item_seed)
        file_id = example.file_id
        microphone = self._read(
            "nearend_mic_signal", f"nearend_mic_fileid_{file_id}.wav"
        )
        render = self._read(
            "farend_speech", f"farend_speech_fileid_{file_id}.wav"
        )
        nearend = self._read(
            "nearend_speech", f"nearend_speech_fileid_{file_id}.wav"
        )
        echo = self._read("echo_signal", f"echo_fileid_{file_id}.wav")
        length = min(len(microphone), len(render), len(nearend), len(echo))
        microphone = microphone[:length]
        render = render[:length]
        nearend = nearend[:length] * example.nearend_scale
        echo = echo[:length]
        residual = microphone - nearend - echo
        activity, _ = nearend_activity(nearend)

        crop_start = 0
        if self.sequence_samples is not None and length > self.sequence_samples:
            maximum_hop = (length - self.sequence_samples) // HOP_LENGTH
            crop_start = rng.randint(0, maximum_hop) * HOP_LENGTH
            crop_end = crop_start + self.sequence_samples
            render = render[crop_start:crop_end]
            nearend = nearend[crop_start:crop_end]
            echo = echo[crop_start:crop_end]
            residual = residual[crop_start:crop_end]
            length = self.sequence_samples

        mode = "mixture"
        if self.augment:
            mode_roll = rng.random()
            if mode_roll < 0.22:
                mode = "far_only"
            elif mode_roll < 0.34:
                mode = "near_only"

            if mode != "near_only" and rng.random() < 0.35:
                echo = _echo_path_movement(echo, rng)

            if mode == "far_only":
                microphone = residual + echo
            elif mode == "near_only":
                microphone = residual + nearend
                render = np.zeros_like(render)
            else:
                microphone = residual + nearend + echo

            if mode != "near_only":
                render = _shift(render, rng.randint(-1_280, 1_280))
                render *= 10.0 ** (rng.uniform(-12.0, 12.0) / 20.0)
            microphone *= 10.0 ** (rng.uniform(-8.0, 8.0) / 20.0)
            if rng.random() < 0.25:
                noise_rms = 10.0 ** (rng.uniform(-55.0, -38.0) / 20.0)
                noise = numpy_rng.normal(0.0, noise_rms, length).astype(np.float32)
                microphone += noise
            if rng.random() < 0.15:
                clip_limit = rng.uniform(0.25, 0.95)
                microphone = np.clip(microphone, -clip_limit, clip_limit)
        else:
            microphone = residual + nearend + echo

        frame_count = 1 + (length - FRAME_LENGTH) // HOP_LENGTH
        label_start = crop_start // HOP_LENGTH + 1
        labels = activity[label_start : label_start + frame_count].copy()
        if len(labels) < frame_count:
            labels = np.pad(labels, (0, frame_count - len(labels)))
        if mode == "far_only":
            labels.fill(0.0)

        microphone = np.nan_to_num(
            microphone, nan=0.0, posinf=1.0, neginf=-1.0
        ).astype(np.float32, copy=False)
        render = np.nan_to_num(
            render, nan=0.0, posinf=1.0, neginf=-1.0
        ).astype(np.float32, copy=False)
        return {
            "microphone": torch.from_numpy(microphone.copy()),
            "render": torch.from_numpy(render.copy()),
            "target": torch.from_numpy(labels),
            "file_id": file_id,
            "ser_db": example.ser_db,
            "farend_nonlinear": example.farend_nonlinear,
            "farend_noisy": example.farend_noisy,
            "nearend_noisy": example.nearend_noisy,
            "mode": mode,
        }
