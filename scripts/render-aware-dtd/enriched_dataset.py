from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfilt
from torch.utils.data import ConcatDataset, Dataset

from aec_dataset import (
    FRAME_LENGTH,
    HOP_LENGTH,
    SAMPLE_RATE,
    _echo_path_movement,
    _shift,
    nearend_activity,
)
from audio_io import read_mono_resampled

DatasetItem = dict[str, torch.Tensor | float | int | str | bool]


@dataclass(frozen=True)
class SynthesisDistribution:
    far_only_probability: float = 0.40
    near_only_probability: float = 0.10
    noise_probability: float = 0.70
    reference_dropout_probability: float = 0.10
    echo_path_movement_probability: float = 0.25

    def validate(self) -> None:
        if not 0.0 <= self.far_only_probability <= 1.0:
            raise ValueError("far_only_probability must be in [0, 1]")
        if not 0.0 <= self.near_only_probability <= 1.0:
            raise ValueError("near_only_probability must be in [0, 1]")
        if self.far_only_probability + self.near_only_probability > 1.0:
            raise ValueError("single-talk probabilities cannot exceed one")
        for name in (
            "noise_probability",
            "reference_dropout_probability",
            "echo_path_movement_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class V5StreamAugmentation:
    """Generic stream hardening; every range is independent of hardware identity."""

    reset_probability: float = 0.0
    hard_far_only_probability: float = 0.0
    maximum_gap_hops: int = 8

    def validate(self) -> None:
        for name in ("reset_probability", "hard_far_only_probability"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.maximum_gap_hops <= 0:
            raise ValueError("maximum_gap_hops must be positive")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        stream = path.open("r", encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"dataset manifest not found: {path}") from error
    with stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"dataset manifest is empty: {path}")
    return rows


def resolve_audio_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def read_audio(path: Path) -> np.ndarray:
    return read_mono_resampled(path, target_sample_rate=SAMPLE_RATE)


def sequence_sample_count(sequence_seconds: float) -> int:
    if sequence_seconds <= 0.0:
        raise ValueError("sequence_seconds must be positive")
    samples = int(sequence_seconds * SAMPLE_RATE) // HOP_LENGTH * HOP_LENGTH
    return max(FRAME_LENGTH, samples)


def frame_count(sample_count: int) -> int:
    if sample_count < FRAME_LENGTH:
        raise ValueError(
            f"audio requires at least {FRAME_LENGTH} samples, got {sample_count}"
        )
    return 1 + (sample_count - FRAME_LENGTH) // HOP_LENGTH


def labels_for(propagated_nearend: np.ndarray) -> np.ndarray:
    """Label frames from the near-end signal that actually reaches the mic."""
    activity, _ = nearend_activity(propagated_nearend)
    count = frame_count(len(propagated_nearend))
    labels = activity[1 : 1 + count].copy()
    if len(labels) < count:
        labels = np.pad(labels, (0, count - len(labels)))
    return labels.astype(np.float32, copy=False)


def active_rms(samples: np.ndarray) -> float:
    if not len(samples):
        return 0.0
    threshold = max(1e-4, float(np.max(np.abs(samples))) * 0.02)
    active = samples[np.abs(samples) >= threshold]
    values = active if len(active) else samples
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def scale_to_rms(samples: np.ndarray, target_rms: float) -> np.ndarray:
    current = active_rms(samples)
    if current <= 1e-8:
        return samples.copy()
    return (samples * (target_rms / current)).astype(np.float32)


def crop_or_place(
    samples: np.ndarray,
    length: int,
    rng: random.Random,
    *,
    random_place: bool,
) -> np.ndarray:
    if not len(samples):
        raise ValueError("cannot crop or place empty audio")
    if len(samples) >= length:
        maximum_hop = (len(samples) - length) // HOP_LENGTH
        start = rng.randint(0, maximum_hop) * HOP_LENGTH
        return samples[start : start + length].copy()
    output = np.zeros(length, dtype=np.float32)
    maximum_hop = (length - len(samples)) // HOP_LENGTH
    start = rng.randint(0, maximum_hop) * HOP_LENGTH if random_place else 0
    output[start : start + len(samples)] = samples
    return output


def compose_farend(
    rows: list[dict[str, object]],
    root: Path,
    length: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = np.zeros(length, dtype=np.float32)
    used: list[dict[str, object]] = []
    cursor = 0
    while cursor < length and len(used) < 3:
        row = rows[rng.randrange(len(rows))]
        clip = read_audio(resolve_audio_path(root, row["path"]))
        if len(clip) > length - cursor:
            available = length - cursor
            maximum_hop = max(0, (len(clip) - available) // HOP_LENGTH)
            start = rng.randint(0, maximum_hop) * HOP_LENGTH
            clip = clip[start : start + available]
        stop = min(length, cursor + len(clip))
        output[cursor:stop] = clip[: stop - cursor]
        used.append(row)
        if stop >= length:
            break
        gap_hops = rng.randint(2, 15)
        cursor = min(length, stop + gap_hops * HOP_LENGTH)
    return output, used


def fft_convolve(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    if not np.any(signal) or not np.any(impulse):
        return np.zeros_like(signal)
    impulse = impulse[:SAMPLE_RATE]
    impulse_energy = float(np.sqrt(np.sum(np.square(impulse, dtype=np.float64))))
    if impulse_energy <= 1e-8:
        return np.zeros_like(signal)
    impulse = impulse / impulse_energy
    output_length = len(signal) + len(impulse) - 1
    fft_length = 1 << (output_length - 1).bit_length()
    output = np.fft.irfft(
        np.fft.rfft(signal, fft_length) * np.fft.rfft(impulse, fft_length),
        fft_length,
    )[: len(signal)]
    return output.astype(np.float32)


def apply_clock_drift(
    samples: np.ndarray,
    parts_per_million: int,
) -> np.ndarray:
    if parts_per_million == 0 or len(samples) < 2:
        return samples.copy()
    source_positions = np.arange(len(samples), dtype=np.float64)
    query = source_positions * (1.0 + parts_per_million / 1_000_000.0)
    return np.interp(
        query,
        source_positions,
        samples,
        left=0.0,
        right=0.0,
    ).astype(np.float32)


def apply_fractional_shift(samples: np.ndarray, amount: float) -> np.ndarray:
    """Shift by a continuous sample offset without using a device delay table."""

    if not math.isfinite(amount):
        raise ValueError("fractional shift must be finite")
    if abs(amount) < 1e-9 or len(samples) < 2:
        return samples.copy()
    positions = np.arange(len(samples), dtype=np.float64)
    return np.interp(
        positions - amount,
        positions,
        samples,
        left=0.0,
        right=0.0,
    ).astype(np.float32)


def apply_quantization(samples: np.ndarray, bits: int) -> np.ndarray:
    if bits < 2 or bits > 24:
        raise ValueError("quantization bits must be in [2, 24]")
    peak = float(np.max(np.abs(samples), initial=0.0))
    if peak <= 1e-8:
        return samples.copy()
    levels = float((1 << (bits - 1)) - 1)
    return (np.round(samples / peak * levels) / levels * peak).astype(np.float32)


def apply_smooth_gain_change(
    samples: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    if len(samples) < 2:
        return samples.copy()
    start_gain = 10.0 ** (rng.uniform(-9.0, 9.0) / 20.0)
    end_gain = 10.0 ** (rng.uniform(-9.0, 9.0) / 20.0)
    envelope = np.linspace(start_gain, end_gain, len(samples), dtype=np.float32)
    return (samples * envelope).astype(np.float32)


def splice_stream_discontinuity(
    signals: tuple[np.ndarray, ...],
    rng: random.Random,
    maximum_gap_hops: int,
) -> tuple[tuple[np.ndarray, ...], int, int]:
    """Remove a gap and return the first valid post-gap frame.

    The removed samples never become zero-render inference frames. Training can
    therefore clear recurrent state immediately before ``reset_frame`` just as
    the production stream does at an epoch/discontinuity boundary.
    """

    if not signals or len({len(signal) for signal in signals}) != 1:
        raise ValueError("discontinuity signals must be nonempty and aligned")
    samples = len(signals[0])
    frames = frame_count(samples)
    earliest = min(5, max(1, frames // 4))
    latest = frames - 5
    if latest <= earliest:
        return tuple(signal.copy() for signal in signals), 0, 0
    reset_frame = rng.randint(earliest, latest)
    available_gap_hops = (
        samples - reset_frame * HOP_LENGTH - FRAME_LENGTH
    ) // HOP_LENGTH
    if available_gap_hops <= 0:
        return tuple(signal.copy() for signal in signals), 0, 0
    gap_hops = rng.randint(1, min(maximum_gap_hops, available_gap_hops))
    start = reset_frame * HOP_LENGTH
    stop = start + gap_hops * HOP_LENGTH
    outputs: list[np.ndarray] = []
    for signal in signals:
        output = np.concatenate(
            (
                signal[:start],
                signal[stop:],
                np.zeros(stop - start, dtype=signal.dtype),
            )
        )
        outputs.append(output.astype(np.float32, copy=False))
    return tuple(outputs), reset_frame, gap_hops


def apply_random_bandwidth(
    samples: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    highpass_hz = rng.uniform(40.0, 280.0)
    lowpass_hz = rng.uniform(max(3_200.0, highpass_hz * 8.0), 7_800.0)
    filters = butter(
        2,
        [highpass_hz, lowpass_hz],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )
    return sosfilt(filters, samples).astype(np.float32)


def apply_memory_nonlinearity(
    samples: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    drive = rng.uniform(1.2, 4.5)
    memory = rng.uniform(-0.20, 0.35)
    driven = samples.copy()
    driven[1:] += memory * samples[:-1]
    denominator = np.tanh(drive)
    return (np.tanh(driven * drive) / denominator).astype(np.float32)


def apply_compressor(
    samples: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    threshold = 10.0 ** (rng.uniform(-24.0, -8.0) / 20.0)
    ratio = rng.uniform(2.0, 8.0)
    magnitude = np.abs(samples)
    compressed = np.where(
        magnitude <= threshold,
        magnitude,
        threshold + (magnitude - threshold) / ratio,
    )
    return (np.sign(samples) * compressed).astype(np.float32)


def apply_reference_dropout(
    render: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    output = render.copy()
    duration_hops = rng.randint(1, 8)
    if duration_hops * HOP_LENGTH >= len(output):
        return output
    start_hop = rng.randint(
        0,
        (len(output) - duration_hops * HOP_LENGTH) // HOP_LENGTH,
    )
    start = start_hop * HOP_LENGTH
    output[start : start + duration_hops * HOP_LENGTH] = 0.0
    return output


def residual_alignment_samples(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.70:
        milliseconds = rng.uniform(-10.0, 10.0)
    elif roll < 0.90:
        milliseconds = rng.uniform(-30.0, 30.0)
    else:
        milliseconds = rng.uniform(-100.0, 100.0)
    return round(milliseconds * SAMPLE_RATE / 1_000.0)


def sanitize(samples: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        samples,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).astype(np.float32, copy=False)


def real_recording_cluster_id(row: dict[str, object]) -> str:
    """Collapse correlated real-recording variants without device conditioning."""

    recording_id = str(row["recording_id"])
    group = row.get("cluster_id", row.get("group"))
    if group is None:
        group = recording_id
        for marker in ("_farend_singletalk", "_nearend"):
            if marker in recording_id:
                group = recording_id.split(marker, 1)[0]
                break
    return f"aec_real:{row['collection']}:{group}"


def standard_item(
    microphone: np.ndarray,
    render: np.ndarray,
    target: np.ndarray,
    *,
    source: str,
    mode: str,
    file_id: str,
    ser_db: float,
    farend_nonlinear: bool,
    farend_noisy: bool,
    nearend_noisy: bool,
    reset_frame: int = 0,
    gap_hops: int = 0,
    cluster_id: str | None = None,
) -> DatasetItem:
    if gap_hops < 0:
        raise ValueError("gap_hops cannot be negative")
    if (reset_frame > 0) != (gap_hops > 0):
        raise ValueError(
            "reset_frame and gap_hops must either both be set or both be zero"
        )
    valid_mask = np.ones(len(target), dtype=bool)
    reset_before = np.zeros(len(target), dtype=bool)
    if len(reset_before):
        reset_before[0] = True
    if reset_frame > 0:
        if reset_frame >= len(reset_before):
            raise ValueError("reset_frame must identify an existing target frame")
        if gap_hops >= len(valid_mask):
            raise ValueError("gap_hops must leave at least one valid target frame")
        # A 640-sample frame at reset_frame-1 would overlap both sides of the
        # removed gap. Production clears its overlap buffer and never emits it.
        valid_mask[reset_frame - 1] = False
        # Fixed-size batches are padded only after the shortened stream. Those
        # trailing frames do not exist at runtime and must not contribute to
        # objectives, event exposure, or confidence intervals.
        valid_mask[-gap_hops:] = False
        reset_before[reset_frame] = True
    stream_epoch = np.cumsum(reset_before, dtype=np.int64) - 1
    return {
        "microphone": torch.from_numpy(sanitize(microphone)),
        "render": torch.from_numpy(sanitize(render)),
        "target": torch.from_numpy(target.astype(np.float32, copy=False)),
        "source": source,
        "mode": mode,
        "file_id": file_id,
        "cluster_id": cluster_id or file_id,
        "ser_db": float(ser_db),
        "farend_nonlinear": farend_nonlinear,
        "farend_noisy": farend_noisy,
        "nearend_noisy": nearend_noisy,
        "valid_mask": torch.from_numpy(valid_mask),
        "reset_before": torch.from_numpy(reset_before),
        "stream_epoch": torch.from_numpy(stream_epoch),
        "gap_hops": int(gap_hops),
        "has_reset": bool(reset_frame > 0),
    }


def choose_mode(rng: random.Random, distribution: SynthesisDistribution) -> str:
    roll = rng.random()
    if roll < distribution.far_only_probability:
        return "far_only"
    if roll < (distribution.far_only_probability + distribution.near_only_probability):
        return "near_only"
    return "mixture"


def augment_stems(
    render: np.ndarray,
    nearend: np.ndarray,
    echo: np.ndarray,
    residual: np.ndarray,
    rng: random.Random,
    numpy_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    mode = choose_mode(
        rng,
        SynthesisDistribution(
            far_only_probability=0.30,
            near_only_probability=0.12,
        ),
    )
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
        render = _shift(render, residual_alignment_samples(rng))
        render *= 10.0 ** (rng.uniform(-8.0, 8.0) / 20.0)
        if rng.random() < 0.08:
            render = apply_reference_dropout(render, rng)
    if rng.random() < 0.35:
        microphone = apply_random_bandwidth(microphone, rng)
    microphone *= 10.0 ** (rng.uniform(-8.0, 8.0) / 20.0)
    if rng.random() < 0.30:
        base_rms = max(active_rms(microphone), 1e-4)
        noise_rms = base_rms / 10.0 ** (rng.uniform(10.0, 35.0) / 20.0)
        microphone += numpy_rng.normal(0.0, noise_rms, len(microphone)).astype(
            np.float32
        )
    if rng.random() < 0.20:
        microphone = apply_compressor(microphone, rng)
    if rng.random() < 0.12:
        clip_limit = rng.uniform(0.25, 0.95)
        microphone = np.clip(microphone, -clip_limit, clip_limit)
    return sanitize(microphone), sanitize(render), mode


class ManifestStemDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        root: str | Path,
        manifest_name: str,
        *,
        source: str,
        sequence_seconds: float,
        augment: bool,
        seed: int,
        limit: int = 0,
    ):
        self.root = Path(root)
        self.rows = load_jsonl(self.root / "manifests" / f"{manifest_name}.jsonl")
        if limit > 0:
            self.rows = self.rows[:limit]
        self.source = source
        self.sequence_samples = sequence_sample_count(sequence_seconds)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read_stems(
        self,
        row: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        microphone = read_audio(resolve_audio_path(self.root, row["microphone"]))
        render = read_audio(resolve_audio_path(self.root, row["render"]))
        nearend = read_audio(resolve_audio_path(self.root, row["nearend"]))
        target = read_audio(resolve_audio_path(self.root, row["target"]))
        echo = read_audio(resolve_audio_path(self.root, row["echo"]))
        scale = float(row.get("nearend_scale", 1.0))
        nearend *= scale
        target *= scale
        return microphone, render, nearend, target, echo

    def __getitem__(self, index: int) -> DatasetItem:
        row = self.rows[index]
        item_seed = self.seed + self.epoch * 1_000_003 + index * 9_176
        rng = random.Random(item_seed)
        numpy_rng = np.random.default_rng(item_seed)
        microphone, render, nearend, target, echo = self._read_stems(row)
        length = min(
            len(microphone),
            len(render),
            len(nearend),
            len(target),
            len(echo),
        )
        if length < FRAME_LENGTH:
            raise ValueError(f"AEC stem set is too short: {row['file_id']}")
        microphone = microphone[:length]
        render = render[:length]
        nearend = nearend[:length]
        target = target[:length]
        echo = echo[:length]
        residual = microphone - nearend - echo

        if length > self.sequence_samples:
            maximum_hop = (length - self.sequence_samples) // HOP_LENGTH
            start = (
                rng.randint(0, maximum_hop) * HOP_LENGTH
                if self.augment
                else maximum_hop // 2 * HOP_LENGTH
            )
            stop = start + self.sequence_samples
            render = render[start:stop]
            nearend = nearend[start:stop]
            target = target[start:stop]
            echo = echo[start:stop]
            residual = residual[start:stop]
        elif length < self.sequence_samples:
            padding = self.sequence_samples - length
            render = np.pad(render, (0, padding))
            nearend = np.pad(nearend, (0, padding))
            target = np.pad(target, (0, padding))
            echo = np.pad(echo, (0, padding))
            residual = np.pad(residual, (0, padding))

        if self.augment:
            microphone, render, mode = augment_stems(
                render,
                nearend,
                echo,
                residual,
                rng,
                numpy_rng,
            )
        else:
            microphone = residual + nearend + echo
            mode = "mixture"
        labels = labels_for(target)
        if mode == "far_only":
            labels.fill(0.0)
        return standard_item(
            microphone,
            render,
            labels,
            source=self.source,
            mode=mode,
            file_id=str(row["file_id"]),
            ser_db=float(row["ser_db"]),
            farend_nonlinear=bool(row.get("farend_nonlinear", False)),
            farend_noisy=bool(row["farend_noisy"]),
            nearend_noisy=bool(row["nearend_noisy"]),
            cluster_id=(
                f"{self.source}:"
                f"{row.get('cluster_id', row.get('group', row['file_id']))}"
            ),
        )


class OnlineSyntheticDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        examples: int,
        sequence_seconds: float,
        seed: int,
        distribution: SynthesisDistribution | None = None,
        stream_augmentation: V5StreamAugmentation | None = None,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split: {split}")
        if examples <= 0:
            raise ValueError("examples must be positive")
        self.root = Path(root)
        manifest_root = self.root / "manifests"
        self.farend = load_jsonl(manifest_root / f"farend_{split}.jsonl")
        self.nearend = load_jsonl(manifest_root / f"nearend_{split}.jsonl")
        self.noise = load_jsonl(manifest_root / f"noise_{split}.jsonl")
        self.rirs = load_jsonl(manifest_root / f"rir_{split}.jsonl")
        self.examples = examples
        self.sequence_samples = sequence_sample_count(sequence_seconds)
        self.seed = seed
        self.epoch = 0
        self.split = split
        self.distribution = distribution or SynthesisDistribution()
        self.distribution.validate()
        self.stream_augmentation = stream_augmentation or V5StreamAugmentation()
        self.stream_augmentation.validate()

    def __len__(self) -> int:
        return self.examples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _choose_audio(
        self,
        rows: list[dict[str, object]],
        rng: random.Random,
        *,
        random_place: bool,
    ) -> tuple[np.ndarray, dict[str, object]]:
        row = rows[rng.randrange(len(rows))]
        samples = read_audio(resolve_audio_path(self.root, row["path"]))
        samples = crop_or_place(
            samples,
            self.sequence_samples,
            rng,
            random_place=random_place,
        )
        return samples, row

    def _choose_rir(
        self,
        rng: random.Random,
    ) -> tuple[np.ndarray, dict[str, object]]:
        row = self.rirs[rng.randrange(len(self.rirs))]
        return read_audio(resolve_audio_path(self.root, row["path"])), row

    def __getitem__(self, index: int) -> DatasetItem:
        item_seed = self.seed + self.epoch * 1_000_003 + index * 65_537
        rng = random.Random(item_seed)
        mode = choose_mode(rng, self.distribution)
        source_parts: list[str] = []

        if mode != "far_only":
            near_dry, near_row = self._choose_audio(
                self.nearend,
                rng,
                random_place=True,
            )
            near_dry = scale_to_rms(
                near_dry,
                10.0 ** (rng.uniform(-30.0, -18.0) / 20.0),
            )
            near_rir, near_rir_row = self._choose_rir(rng)
            propagated_nearend = fft_convolve(near_dry, near_rir)
            propagated_nearend = scale_to_rms(
                propagated_nearend,
                max(active_rms(near_dry), 1e-5),
            )
            source_parts.extend((str(near_row["corpus"]), str(near_rir_row["corpus"])))
        else:
            propagated_nearend = np.zeros(
                self.sequence_samples,
                dtype=np.float32,
            )

        farend_nonlinear = False
        if mode != "near_only":
            render, far_rows = compose_farend(
                self.farend,
                self.root,
                self.sequence_samples,
                rng,
            )
            render = scale_to_rms(
                render,
                10.0 ** (rng.uniform(-28.0, -14.0) / 20.0),
            )
            drift_ppm = rng.randint(-180, 180)
            echo_input = apply_clock_drift(render, drift_ppm)
            if rng.random() < 0.75:
                echo_input = apply_random_bandwidth(echo_input, rng)
            if rng.random() < 0.50:
                echo_input = apply_memory_nonlinearity(echo_input, rng)
                farend_nonlinear = True
            if rng.random() < 0.35:
                echo_input = apply_compressor(echo_input, rng)

            echo_rir, echo_rir_row = self._choose_rir(rng)
            echo = fft_convolve(echo_input, echo_rir)
            physical_delay = rng.randint(0, round(0.30 * SAMPLE_RATE))
            echo = _shift(echo, physical_delay)
            render = _shift(
                render,
                physical_delay + residual_alignment_samples(rng),
            )
            if mode == "mixture":
                ser_db = (
                    rng.uniform(-18.0, 3.0)
                    if rng.random() < 0.75
                    else rng.uniform(3.0, 15.0)
                )
                desired_echo_rms = active_rms(propagated_nearend) / (
                    10.0 ** (ser_db / 20.0)
                )
            else:
                ser_db = -99.0
                desired_echo_rms = 10.0 ** (rng.uniform(-28.0, -10.0) / 20.0)
            echo = scale_to_rms(echo, max(desired_echo_rms, 1e-6))
            if rng.random() < self.distribution.echo_path_movement_probability:
                echo = _echo_path_movement(echo, rng)
            if (
                mode == "far_only"
                and rng.random() < self.stream_augmentation.hard_far_only_probability
            ):
                # The detector must recognize echo even after generic path,
                # clock, gain, quantization, and weak-reference corruption.
                echo = apply_smooth_gain_change(echo, rng)
                echo = apply_memory_nonlinearity(echo, rng)
                if rng.random() < 0.70:
                    echo = apply_compressor(echo, rng)
                if rng.random() < 0.65:
                    echo = apply_quantization(echo, rng.randint(6, 14))
                if rng.random() < 0.50:
                    clip_limit = rng.uniform(0.15, 0.85)
                    echo = np.clip(echo, -clip_limit, clip_limit)
                render = apply_fractional_shift(
                    render,
                    rng.uniform(-1.0, 1.0),
                )
                render *= 10.0 ** (rng.uniform(-18.0, 6.0) / 20.0)
                farend_nonlinear = True
            if rng.random() < self.distribution.reference_dropout_probability:
                render = apply_reference_dropout(render, rng)
            source_parts.extend(str(row["corpus"]) for row in far_rows)
            source_parts.append(str(echo_rir_row["corpus"]))
        else:
            render = np.zeros(self.sequence_samples, dtype=np.float32)
            echo = render.copy()
            ser_db = 99.0

        microphone = propagated_nearend + echo
        nearend_noisy = False
        if rng.random() < self.distribution.noise_probability:
            noise, noise_row = self._choose_audio(
                self.noise,
                rng,
                random_place=False,
            )
            base_rms = max(active_rms(microphone), 1e-4)
            noise_rms = base_rms / (10.0 ** (rng.uniform(5.0, 32.0) / 20.0))
            microphone += scale_to_rms(noise, noise_rms)
            nearend_noisy = mode != "far_only"
            source_parts.append(str(noise_row["corpus"]))
        if rng.random() < 0.50:
            microphone = apply_random_bandwidth(microphone, rng)
        if rng.random() < 0.30:
            microphone = apply_compressor(microphone, rng)
        render *= 10.0 ** (rng.uniform(-8.0, 8.0) / 20.0)
        microphone *= 10.0 ** (rng.uniform(-8.0, 8.0) / 20.0)
        if rng.random() < 0.15:
            limit = rng.uniform(0.20, 0.95)
            microphone = np.clip(microphone, -limit, limit)

        reset_frame = 0
        gap_hops = 0
        if rng.random() < self.stream_augmentation.reset_probability:
            (
                (microphone, render, propagated_nearend),
                reset_frame,
                gap_hops,
            ) = splice_stream_discontinuity(
                (microphone, render, propagated_nearend),
                rng,
                self.stream_augmentation.maximum_gap_hops,
            )

        labels = labels_for(propagated_nearend)
        if mode == "far_only":
            labels.fill(0.0)
        if mode == "far_only":
            primary_far = far_rows[0]
            cluster_id = (
                f"far:{primary_far['corpus']}:"
                f"{primary_far.get('group', primary_far['file_id'])}"
            )
        else:
            cluster_id = (
                f"near:{near_row['corpus']}:"
                f"{near_row.get('group', near_row['file_id'])}"
            )
        return standard_item(
            microphone,
            render,
            labels,
            source=f"online_{self.split}",
            mode=mode,
            file_id=(
                f"online-{self.split}-{self.epoch}-{index}:" + ",".join(source_parts)
            ),
            ser_db=ser_db,
            farend_nonlinear=farend_nonlinear,
            farend_noisy=False,
            nearend_noisy=nearend_noisy,
            reset_frame=reset_frame,
            gap_hops=gap_hops,
            cluster_id=cluster_id,
        )


class RealSingleTalkDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        root: str | Path,
        manifest_name: str,
        *,
        sequence_seconds: float,
        augment: bool,
        seed: int,
        limit: int = 0,
    ):
        self.root = Path(root)
        self.rows = load_jsonl(self.root / "manifests" / f"{manifest_name}.jsonl")
        if limit > 0:
            self.rows = self.rows[:limit]
        self.sequence_samples = sequence_sample_count(sequence_seconds)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> DatasetItem:
        row = self.rows[index]
        item_seed = self.seed + self.epoch * 1_000_003 + index * 1_009
        rng = random.Random(item_seed)
        microphone = read_audio(resolve_audio_path(self.root, row["microphone"]))
        mode = str(row["mode"])
        render_value = str(row.get("render", ""))
        if render_value:
            render = read_audio(resolve_audio_path(self.root, render_value))
            length = min(len(microphone), len(render))
            microphone = microphone[:length]
            render = render[:length]
        else:
            render = np.zeros_like(microphone)

        if len(microphone) >= self.sequence_samples:
            maximum_hop = (len(microphone) - self.sequence_samples) // HOP_LENGTH
            start = (
                rng.randint(0, maximum_hop) * HOP_LENGTH
                if self.augment
                else maximum_hop // 2 * HOP_LENGTH
            )
            stop = start + self.sequence_samples
            microphone = microphone[start:stop]
            render = render[start:stop]
        else:
            padding = self.sequence_samples - len(microphone)
            microphone = np.pad(microphone, (0, padding))
            render = np.pad(render, (0, padding))

        if self.augment:
            if mode == "far_only":
                render = _shift(render, residual_alignment_samples(rng))
                render *= 10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)
            if rng.random() < 0.35:
                microphone = apply_random_bandwidth(microphone, rng)
            microphone *= 10.0 ** (rng.uniform(-4.0, 4.0) / 20.0)
        labels = (
            np.zeros(frame_count(self.sequence_samples), dtype=np.float32)
            if mode == "far_only"
            else labels_for(microphone)
        )
        return standard_item(
            microphone,
            render,
            labels,
            source="aec_real:" + str(row["collection"]),
            mode=mode,
            file_id=str(row["recording_id"]),
            ser_db=-99.0 if mode == "far_only" else 99.0,
            farend_nonlinear=False,
            farend_noisy=False,
            nearend_noisy=False,
            cluster_id=real_recording_cluster_id(row),
        )


class WeightedMixedDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        datasets: list[Dataset[DatasetItem]],
        weights: list[float],
        *,
        examples: int,
        seed: int,
    ):
        if len(datasets) != len(weights) or not datasets:
            raise ValueError("datasets and weights must have the same nonzero length")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("mixed datasets must not be empty")
        if not all(math.isfinite(weight) for weight in weights):
            raise ValueError("mixed dataset weights must be finite")
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("mixed dataset weights must sum to a positive value")
        self.datasets = datasets
        self.cumulative = np.cumsum(np.asarray(weights, dtype=np.float64) / total)
        self.examples = examples
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.examples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for dataset in self.datasets:
            setter = getattr(dataset, "set_epoch", None)
            if setter is not None:
                setter(epoch)

    def __getitem__(self, index: int) -> DatasetItem:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index * 104_729)
        source_index = int(np.searchsorted(self.cumulative, rng.random(), side="right"))
        source_index = min(source_index, len(self.datasets) - 1)
        dataset = self.datasets[source_index]
        return dataset[rng.randrange(len(dataset))]


def create_enriched_datasets(
    root: str | Path,
    *,
    sequence_seconds: float,
    examples_per_epoch: int,
    validation_examples: int,
    seed: int,
    dataset_weights: tuple[float, float, float, float] = (
        0.45,
        0.25,
        0.25,
        0.05,
    ),
    wenet_far_only_probability: float = 0.40,
    v5_reset_probability: float = 0.0,
    v5_hard_far_only_probability: float = 0.0,
    v5_maximum_gap_hops: int = 8,
    v5_validation_reset_probability: float = 0.2,
    v5_validation_hard_far_only_probability: float = 0.5,
    v5_validation_maximum_gap_hops: int = 8,
    train_limit: int = 0,
    validation_limit: int = 0,
) -> tuple[WeightedMixedDataset, Dataset[DatasetItem], dict[str, object]]:
    if len(dataset_weights) != 4:
        raise ValueError(
            "dataset_weights must contain narrow, fullband, Wenet, and real weights"
        )
    if not all(math.isfinite(weight) for weight in dataset_weights):
        raise ValueError("dataset_weights must be finite")
    if any(weight < 0.0 for weight in dataset_weights):
        raise ValueError("dataset_weights must be non-negative")
    if sum(dataset_weights) <= 0.0:
        raise ValueError("dataset_weights must sum to a positive value")
    online_train_distribution = SynthesisDistribution(
        far_only_probability=wenet_far_only_probability,
    )
    online_train_distribution.validate()
    train_stream_augmentation = V5StreamAugmentation(
        reset_probability=v5_reset_probability,
        hard_far_only_probability=v5_hard_far_only_probability,
        maximum_gap_hops=v5_maximum_gap_hops,
    )
    train_stream_augmentation.validate()
    validation_stream_augmentation = V5StreamAugmentation(
        reset_probability=v5_validation_reset_probability,
        hard_far_only_probability=v5_validation_hard_far_only_probability,
        maximum_gap_hops=v5_validation_maximum_gap_hops,
    )
    validation_stream_augmentation.validate()
    narrow_train = ManifestStemDataset(
        root,
        "narrow_train",
        source="aec_narrow",
        sequence_seconds=sequence_seconds,
        augment=True,
        seed=seed,
        limit=train_limit,
    )
    fullband_train = ManifestStemDataset(
        root,
        "fullband_train",
        source="aec_fullband",
        sequence_seconds=sequence_seconds,
        augment=True,
        seed=seed + 1,
        limit=train_limit,
    )
    online_train = OnlineSyntheticDataset(
        root,
        split="train",
        examples=max(1, examples_per_epoch),
        sequence_seconds=sequence_seconds,
        seed=seed + 2,
        distribution=online_train_distribution,
        stream_augmentation=train_stream_augmentation,
    )
    real_train = RealSingleTalkDataset(
        root,
        "real_train",
        sequence_seconds=sequence_seconds,
        augment=True,
        seed=seed + 3,
        limit=train_limit,
    )
    weights = list(dataset_weights)
    train = WeightedMixedDataset(
        [narrow_train, fullband_train, online_train, real_train],
        weights,
        examples=examples_per_epoch,
        seed=seed,
    )

    narrow_validation = ManifestStemDataset(
        root,
        "narrow_validation",
        source="aec_narrow",
        sequence_seconds=sequence_seconds,
        augment=False,
        seed=seed,
        limit=validation_limit,
    )
    fullband_validation = ManifestStemDataset(
        root,
        "fullband_validation",
        source="aec_fullband",
        sequence_seconds=sequence_seconds,
        augment=False,
        seed=seed,
        limit=validation_limit,
    )
    real_validation = RealSingleTalkDataset(
        root,
        "real_validation",
        sequence_seconds=sequence_seconds,
        augment=False,
        seed=seed,
        limit=validation_limit,
    )
    online_validation = OnlineSyntheticDataset(
        root,
        split="validation",
        examples=validation_examples,
        sequence_seconds=sequence_seconds,
        seed=seed + 4,
        distribution=SynthesisDistribution(
            far_only_probability=0.50,
            near_only_probability=0.10,
        ),
        stream_augmentation=validation_stream_augmentation,
    )
    validation: Dataset[DatasetItem] = ConcatDataset(
        [
            narrow_validation,
            fullband_validation,
            real_validation,
            online_validation,
        ]
    )
    description = {
        "schema_version": "2.0",
        "weights": {
            "aec_narrow": weights[0],
            "aec_fullband": weights[1],
            "wenet_online_synthetic": weights[2],
            "aec_real_single_talk": weights[3],
        },
        "train_source_examples": {
            "aec_narrow": len(narrow_train),
            "aec_fullband": len(fullband_train),
            "wenet_online_synthetic_budget": len(online_train),
            "aec_real_single_talk": len(real_train),
        },
        "examples_per_epoch": len(train),
        "validation_examples": len(validation),
        "sequence_seconds": sequence_seconds,
        "farend_corpus": "WenetSpeech4TTS only for online synthesis",
        "online_train_distribution": {
            "far_only_probability": (online_train_distribution.far_only_probability),
            "near_only_probability": (online_train_distribution.near_only_probability),
            "noise_probability": online_train_distribution.noise_probability,
            "reference_dropout_probability": (
                online_train_distribution.reference_dropout_probability
            ),
            "echo_path_movement_probability": (
                online_train_distribution.echo_path_movement_probability
            ),
        },
        "v5_train_stream_augmentation": {
            "reset_probability": train_stream_augmentation.reset_probability,
            "hard_far_only_probability": (
                train_stream_augmentation.hard_far_only_probability
            ),
            "maximum_gap_hops": train_stream_augmentation.maximum_gap_hops,
            "device_conditioning": False,
            "gap_contract": "gap samples omitted; recurrent state reset before next valid frame",
        },
        "v5_validation_stream_augmentation": {
            "reset_probability": validation_stream_augmentation.reset_probability,
            "hard_far_only_probability": (
                validation_stream_augmentation.hard_far_only_probability
            ),
            "maximum_gap_hops": validation_stream_augmentation.maximum_gap_hops,
            "device_conditioning": False,
            "gap_contract": "gap samples omitted; recurrent state reset before next valid frame",
        },
        "runtime_resampling": "per-file header rate -> 16 kHz mono",
    }
    return train, validation, description


def create_evaluation_suites(
    root: str | Path,
    *,
    sequence_seconds: float,
    online_examples: int,
    seed: int,
    limit: int = 0,
) -> dict[str, Dataset[DatasetItem]]:
    return {
        "aec_official": ManifestStemDataset(
            root,
            "narrow_test",
            source="aec_narrow_official_test",
            sequence_seconds=sequence_seconds,
            augment=False,
            seed=seed,
            limit=limit,
        ),
        "wenet_unseen": OnlineSyntheticDataset(
            root,
            split="test",
            examples=online_examples,
            sequence_seconds=sequence_seconds,
            seed=seed + 1,
            distribution=SynthesisDistribution(
                far_only_probability=0.55,
                near_only_probability=0.10,
            ),
        ),
        "aec_real_unseen": RealSingleTalkDataset(
            root,
            "real_test",
            sequence_seconds=sequence_seconds,
            augment=False,
            seed=seed + 2,
            limit=limit,
        ),
    }
