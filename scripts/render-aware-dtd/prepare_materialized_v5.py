from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from aec_dataset import HOP_LENGTH, SAMPLE_RATE, _shift
from enriched_dataset import (
    active_rms,
    apply_clock_drift,
    apply_compressor,
    apply_fractional_shift,
    apply_memory_nonlinearity,
    apply_quantization,
    apply_random_bandwidth,
    apply_reference_dropout,
    apply_smooth_gain_change,
    compose_farend,
    crop_or_place,
    fft_convolve,
    labels_for,
    load_jsonl,
    read_audio,
    residual_alignment_samples,
    sanitize,
    scale_to_rms,
)

SEQUENCE_SECONDS = 8
SEQUENCE_SAMPLES = SAMPLE_RATE * SEQUENCE_SECONDS
BASE_MANIFEST_NAMES = (
    "narrow_train",
    "narrow_validation",
    "narrow_test",
    "fullband_train",
    "fullband_validation",
    "fullband_test",
    "farend_train",
    "farend_validation",
    "farend_test",
    "nearend_train",
    "nearend_validation",
    "nearend_test",
    "noise_train",
    "noise_validation",
    "noise_test",
    "rir_train",
    "rir_validation",
    "rir_test",
    "real_train",
    "real_validation",
    "real_test",
)
EXCLUDED_KWS_CORPORA = (
    "AISHELL-WakeUp-1-sample",
    "mobvoi_hotword_dataset",
    "speech_asr_aishell_hotwords_testsets",
    "speech_kws_mobile_xiaoyun_pos_testsets",
)
ROOM_SPLITS = {
    "VUT_FIT_D105": "train",
    "VUT_FIT_L207": "train",
    "VUT_FIT_L212": "train",
    "VUT_FIT_Q301": "validation",
    "VUT_FIT_L227": "test",
}


@dataclass(frozen=True)
class V5MaterializationConfig:
    seed: int = 43
    train_examples: int = 48_000
    validation_examples: int = 1_500
    reset_probability: float = 0.30
    movement_probability: float = 0.50
    noise_probability: float = 0.75
    render_gap_probability: float = 0.20
    real_echo_probability: float = 0.50
    hard_far_probability: float = 0.60
    maximum_gap_hops: int = 8


@dataclass(frozen=True)
class SourcePools:
    root: Path
    farend: list[dict[str, object]]
    nearend: list[dict[str, object]]
    noise: list[dict[str, object]]
    rirs: list[dict[str, object]]
    real_near: list[dict[str, object]]
    real_far_static: list[dict[str, object]]
    real_far_movement: list[dict[str, object]]
    but_by_utterance: dict[str, list[dict[str, object]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a materialized render-aware DTD v5 dataset."
    )
    parser.add_argument(
        "--v4-root",
        default=r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4",
    )
    parser.add_argument(
        "--but-replayed-root",
        default=(
            r"E:\Projects\render-aware-dtd\datasets"
            r"\BUT_ReverbDB_rel_19_06_LibriSpeech-Only"
        ),
    )
    parser.add_argument(
        "--target-root",
        default=r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v5",
    )
    parser.add_argument("--train-examples", type=int, default=48_000)
    parser.add_argument("--validation-examples", type=int, default=1_500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def stable_speaker_splits(speakers: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(
        speakers,
        key=lambda speaker: hashlib.sha256(
            f"but-libri:{seed}:{speaker}".encode("utf-8")
        ).digest(),
    )
    test_count = max(1, round(len(ordered) * 0.10))
    validation_count = max(1, round(len(ordered) * 0.10))
    assignments: dict[str, str] = {}
    for speaker in ordered[:test_count]:
        assignments[speaker] = "test"
    for speaker in ordered[test_count : test_count + validation_count]:
        assignments[speaker] = "validation"
    for speaker in ordered[test_count + validation_count :]:
        assignments[speaker] = "train"
    return assignments


def parse_but_replayed_path(root: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) < 10 or "LibriSpeech" not in parts:
        raise ValueError(f"unexpected BUT retransmission path: {path}")
    base_utterance = re.sub(r"\.v\d\d$", "", path.stem)
    tokens = base_utterance.split("-")
    if len(tokens) < 3:
        raise ValueError(f"cannot parse LibriSpeech utterance id: {path.name}")
    return {
        "room": parts[0],
        "mic_setup": parts[1],
        "speaker_setup": parts[2],
        "channel": parts[3],
        "speaker": tokens[0],
        "utterance_id": base_utterance,
    }


def discover_but_replayed(
    root: Path,
    seed: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    channel_roots = sorted(
        path
        for path in root.glob("*/*/*/*/english/LibriSpeech/test-clean")
        if path.is_dir()
    )
    if not channel_roots:
        raise FileNotFoundError(f"no BUT retransmission channels found: {root}")
    speaker_directories = sorted(
        path for path in channel_roots[0].iterdir() if path.is_dir()
    )
    speakers = [path.name for path in speaker_directories]
    assignments = stable_speaker_splits(speakers, seed)
    utterances_per_speaker = {
        path.name: sum(1 for _ in path.rglob("*.wav")) for path in speaker_directories
    }
    views_per_room = Counter(path.relative_to(root).parts[0] for path in channel_roots)
    manifests: dict[str, list[dict[str, object]]] = defaultdict(list)
    for channel_root in channel_roots:
        room = channel_root.relative_to(root).parts[0]
        room_split = ROOM_SPLITS.get(room)
        if room_split is None:
            continue
        selected_speakers = [
            speaker for speaker in speakers if assignments[speaker] == room_split
        ]
        for speaker in selected_speakers:
            for path in sorted((channel_root / speaker).rglob("*.wav")):
                metadata = parse_but_replayed_path(root, path)
                manifests[room_split].append(
                    {
                        "path": str(path.resolve()),
                        "file_id": path.stem,
                        "utterance_id": metadata["utterance_id"],
                        "group": metadata["speaker"],
                        "speaker": metadata["speaker"],
                        "room": metadata["room"],
                        "mic_setup": metadata["mic_setup"],
                        "speaker_setup": metadata["speaker_setup"],
                        "channel": metadata["channel"],
                        "corpus": "but_replayed_librispeech",
                    }
                )
    source_wavs = sum(utterances_per_speaker.values()) * len(channel_roots)
    selected_wavs = sum(len(rows) for rows in manifests.values())
    summary = {
        "source_wavs": source_wavs,
        "source_channels": len(channel_roots),
        "source_views_per_room": dict(views_per_room),
        "source_utterances_per_speaker": utterances_per_speaker,
        "selected_wavs": {
            split: len(manifests[split]) for split in ROOM_SPLITS.values()
        },
        "selected_unique_utterances": {
            split: len({str(row["utterance_id"]) for row in manifests[split]})
            for split in ROOM_SPLITS.values()
        },
        "selected_speakers": {
            split: sorted({str(row["speaker"]) for row in manifests[split]})
            for split in ROOM_SPLITS.values()
        },
        "selected_rooms": {
            split: sorted({str(row["room"]) for row in manifests[split]})
            for split in ROOM_SPLITS.values()
        },
        "dropped_cross_block_views": source_wavs - selected_wavs,
        "split_rule": "speaker and room must both belong to the same split",
    }
    return manifests, summary


def copy_v4_manifests(v4_root: Path, target_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    target_manifests = target_root / "manifests"
    target_manifests.mkdir(parents=True, exist_ok=True)
    for name in BASE_MANIFEST_NAMES:
        source = v4_root / "manifests" / f"{name}.jsonl"
        target = target_manifests / source.name
        if not source.is_file():
            raise FileNotFoundError(f"v4 manifest not found: {source}")
        source_hash = sha256_file(source)
        if target.is_file() and sha256_file(target) != source_hash:
            raise ValueError(f"existing v5 base manifest differs from v4: {target}")
        if not target.is_file():
            shutil.copy2(source, target)
        hashes[source.name] = source_hash
    return hashes


def group_but_rows(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["utterance_id"])].append(row)
    return dict(grouped)


def load_source_pools(root: Path, split: str) -> SourcePools:
    manifests = root / "manifests"
    real = load_jsonl(manifests / f"real_{split}.jsonl")
    real_near = [row for row in real if row["mode"] == "near_only"]
    real_far = [row for row in real if row["mode"] == "far_only"]
    real_far_movement = [
        row for row in real_far if "movement" in str(row["recording_id"])
    ]
    real_far_static = [
        row for row in real_far if "movement" not in str(row["recording_id"])
    ]
    but_rows = load_jsonl(manifests / f"but_replayed_{split}.jsonl")
    pools = SourcePools(
        root=root,
        farend=load_jsonl(manifests / f"farend_{split}.jsonl"),
        nearend=load_jsonl(manifests / f"nearend_{split}.jsonl"),
        noise=load_jsonl(manifests / f"noise_{split}.jsonl"),
        rirs=load_jsonl(manifests / f"rir_{split}.jsonl"),
        real_near=real_near,
        real_far_static=real_far_static,
        real_far_movement=real_far_movement,
        but_by_utterance=group_but_rows(but_rows),
    )
    if not all(
        (
            pools.farend,
            pools.nearend,
            pools.noise,
            pools.rirs,
            pools.real_near,
            pools.real_far_static,
            pools.real_far_movement,
            pools.but_by_utterance,
        )
    ):
        raise ValueError(f"one or more v5 source pools are empty for {split}")
    return pools


def choose_mode(rng: random.Random, split: str) -> str:
    roll = rng.random()
    far_probability = 0.40 if split == "train" else 0.50
    if roll < far_probability:
        return "far_only"
    if roll < far_probability + 0.10:
        return "near_only"
    return "mixture"


def trim_speech(samples: np.ndarray) -> np.ndarray:
    threshold = max(1e-4, float(np.max(np.abs(samples), initial=0.0)) * 0.02)
    active = np.flatnonzero(np.abs(samples) >= threshold)
    if not len(active):
        return samples
    margin = round(0.08 * SAMPLE_RATE)
    start = max(0, int(active[0]) - margin)
    stop = min(len(samples), int(active[-1]) + margin + 1)
    return samples[start:stop]


def pcm16_roundtrip(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    quantized = np.clip(np.round(clipped * 32_768.0), -32_768.0, 32_767.0)
    return (quantized / 32_768.0).astype(np.float32)


def place_at(samples: np.ndarray, length: int, start: int) -> np.ndarray:
    output = np.zeros(length, dtype=np.float32)
    if start >= length:
        return output
    source_start = max(0, -start)
    destination_start = max(0, start)
    count = min(len(samples) - source_start, length - destination_start)
    if count > 0:
        output[destination_start : destination_start + count] = samples[
            source_start : source_start + count
        ]
    return output


def common_crop_pair(
    microphone: np.ndarray,
    render: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(microphone), len(render))
    microphone = microphone[:length]
    render = render[:length]
    if length >= SEQUENCE_SAMPLES:
        maximum_hop = (length - SEQUENCE_SAMPLES) // HOP_LENGTH
        last_pair: tuple[np.ndarray, np.ndarray] | None = None
        for _ in range(8):
            start = rng.randint(0, maximum_hop) * HOP_LENGTH
            stop = start + SEQUENCE_SAMPLES
            last_pair = microphone[start:stop].copy(), render[start:stop].copy()
            if active_rms(last_pair[0]) > 1e-5 and active_rms(last_pair[1]) > 1e-5:
                return last_pair
        if last_pair is not None:
            return last_pair
    padding = SEQUENCE_SAMPLES - length
    return np.pad(microphone, (0, padding)), np.pad(render, (0, padding))


def choose_near_source(
    pools: SourcePools,
    rng: random.Random,
    reset_frame: int,
    reset_variant: str,
) -> tuple[np.ndarray, dict[str, object]]:
    roll = rng.random()
    if roll < 0.35:
        source_type = "asr_rir"
        row = pools.nearend[rng.randrange(len(pools.nearend))]
        source = read_audio(Path(str(row["path"])))
        source = trim_speech(source)
        source = crop_or_place(source, SEQUENCE_SAMPLES, rng, random_place=True)
        rir_row = pools.rirs[rng.randrange(len(pools.rirs))]
        propagated = fft_convolve(source, read_audio(Path(str(rir_row["path"]))))
        metadata: dict[str, object] = {
            "type": source_type,
            "path": str(row["path"]),
            "file_id": str(row["file_id"]),
            "group": str(row.get("group", row["file_id"])),
            "rir": str(rir_row["path"]),
        }
    elif roll < 0.70:
        source_type = "but_replayed"
        utterance = rng.choice(sorted(pools.but_by_utterance))
        row = rng.choice(pools.but_by_utterance[utterance])
        source = trim_speech(read_audio(Path(str(row["path"]))))
        propagated = crop_or_place(source, SEQUENCE_SAMPLES, rng, random_place=True)
        metadata = {
            "type": source_type,
            "path": str(row["path"]),
            "file_id": str(row["file_id"]),
            "group": str(row["speaker"]),
            "utterance_id": str(row["utterance_id"]),
            "room": str(row["room"]),
            "channel": str(row["channel"]),
        }
    else:
        source_type = "aec_real_near"
        row = pools.real_near[rng.randrange(len(pools.real_near))]
        source = trim_speech(read_audio(Path(str(row["microphone"]))))
        propagated = crop_or_place(source, SEQUENCE_SAMPLES, rng, random_place=True)
        metadata = {
            "type": source_type,
            "path": str(row["microphone"]),
            "file_id": str(row["recording_id"]),
            "group": str(row["recording_id"]).split("_nearend", 1)[0],
            "collection": str(row["collection"]),
        }

    if reset_frame > 0 and reset_variant == "cold_onset":
        compact = trim_speech(source)
        onset_delay_hops = rng.randint(0, 18)
        onset = reset_frame * HOP_LENGTH + onset_delay_hops * HOP_LENGTH
        if source_type == "asr_rir":
            dry = place_at(compact, SEQUENCE_SAMPLES, onset)
            propagated = fft_convolve(dry, read_audio(Path(str(metadata["rir"]))))
        else:
            propagated = place_at(compact, SEQUENCE_SAMPLES, onset)
        metadata["post_reset_onset_delay_ms"] = onset_delay_hops * 20
    target_rms = 10.0 ** (rng.uniform(-30.0, -18.0) / 20.0)
    propagated = scale_to_rms(propagated, target_rms)
    if reset_frame > 0 and reset_variant == "cold_onset":
        propagated = pcm16_roundtrip(propagated)
        requested_delay_hops = int(metadata["post_reset_onset_delay_ms"]) // 20
        desired_frame = reset_frame + requested_delay_hops
        alignment_hops = 0
        for _ in range(4):
            active_frames = np.flatnonzero(labels_for(propagated) > 0.5)
            if not len(active_frames):
                raise ValueError(
                    f"cold-onset near source has no active labels: {metadata}"
                )
            correction_hops = desired_frame - int(active_frames[0])
            if correction_hops == 0:
                break
            alignment_hops += correction_hops
            propagated = _shift(propagated, correction_hops * HOP_LENGTH)
        aligned_active = np.flatnonzero(labels_for(propagated) > 0.5)
        if not len(aligned_active):
            raise ValueError(
                f"aligned cold-onset source has no active labels: {metadata}"
            )
        if int(aligned_active[0]) != desired_frame:
            raise ValueError(
                "cold-onset label alignment did not converge: "
                f"desired={desired_frame}, actual={int(aligned_active[0])}"
            )
        metadata["post_reset_onset_delay_ms"] = (
            int(aligned_active[0]) - reset_frame
        ) * 20
        metadata["label_alignment_hops"] = alignment_hops
    return propagated, metadata


def moving_rir_echo(
    echo_input: np.ndarray,
    pools: SourcePools,
    rng: random.Random,
) -> tuple[np.ndarray, list[str]]:
    first = pools.rirs[rng.randrange(len(pools.rirs))]
    second = pools.rirs[rng.randrange(len(pools.rirs))]
    echo_a = fft_convolve(echo_input, read_audio(Path(str(first["path"]))))
    echo_b = fft_convolve(echo_input, read_audio(Path(str(second["path"]))))
    change = rng.randint(SEQUENCE_SAMPLES // 4, SEQUENCE_SAMPLES * 3 // 4)
    crossfade = min(10 * HOP_LENGTH, change, SEQUENCE_SAMPLES - change)
    start = change - crossfade // 2
    stop = start + crossfade
    output = echo_a.copy()
    output[stop:] = echo_b[stop:]
    blend = np.linspace(0.0, 1.0, crossfade, dtype=np.float32)
    output[start:stop] = echo_a[start:stop] * (1.0 - blend) + echo_b[start:stop] * blend
    return output, [str(first["path"]), str(second["path"])]


def choose_far_source(
    pools: SourcePools,
    rng: random.Random,
    *,
    movement: bool,
    use_real_echo: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], bool, bool]:
    if use_real_echo:
        candidates = pools.real_far_movement if movement else pools.real_far_static
        row = candidates[rng.randrange(len(candidates))]
        echo, render = common_crop_pair(
            read_audio(Path(str(row["microphone"]))),
            read_audio(Path(str(row["render"]))),
            rng,
        )
        render = _shift(render, residual_alignment_samples(rng))
        return (
            echo,
            render,
            {
                "type": "aec_real_echo",
                "microphone": str(row["microphone"]),
                "render": str(row["render"]),
                "file_id": str(row["recording_id"]),
                "group": str(row["recording_id"]).split("_farend", 1)[0],
                "collection": str(row["collection"]),
            },
            False,
            True,
        )

    render, rows = compose_farend(pools.farend, pools.root, SEQUENCE_SAMPLES, rng)
    render = scale_to_rms(render, 10.0 ** (rng.uniform(-28.0, -14.0) / 20.0))
    echo_input = apply_clock_drift(render, rng.randint(-180, 180))
    if rng.random() < 0.75:
        echo_input = apply_random_bandwidth(echo_input, rng)
    nonlinear = rng.random() < 0.50
    if nonlinear:
        echo_input = apply_memory_nonlinearity(echo_input, rng)
    if rng.random() < 0.35:
        echo_input = apply_compressor(echo_input, rng)
    if movement:
        echo, rir_paths = moving_rir_echo(echo_input, pools, rng)
    else:
        rir = pools.rirs[rng.randrange(len(pools.rirs))]
        echo = fft_convolve(echo_input, read_audio(Path(str(rir["path"]))))
        rir_paths = [str(rir["path"])]
    physical_delay = rng.randint(0, round(0.30 * SAMPLE_RATE))
    echo = _shift(echo, physical_delay)
    render = _shift(render, physical_delay + residual_alignment_samples(rng))
    return (
        echo,
        render,
        {
            "type": "wenet_rir_echo",
            "paths": [str(row["path"]) for row in rows],
            "file_ids": [str(row["file_id"]) for row in rows],
            "group": str(rows[0].get("group", rows[0]["file_id"])),
            "rirs": rir_paths,
            "physical_delay_samples": physical_delay,
        },
        nonlinear,
        False,
    )


def apply_device_profile(
    microphone: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, str]:
    roll = rng.random()
    if roll < 0.35:
        profile = "mobile_resampled"
        intermediate_rate = rng.choice((8_000, 12_000))
        microphone = resample_poly(microphone, intermediate_rate, SAMPLE_RATE).astype(
            np.float32
        )
        microphone = resample_poly(microphone, SAMPLE_RATE, intermediate_rate).astype(
            np.float32
        )
        microphone = crop_or_place(
            microphone, SEQUENCE_SAMPLES, rng, random_place=False
        )
        microphone = apply_compressor(microphone, rng)
        microphone = apply_quantization(microphone, rng.randint(10, 15))
    elif roll < 0.55:
        profile = "usb_agc"
        microphone = apply_smooth_gain_change(microphone, rng)
        microphone = apply_compressor(microphone, rng)
    else:
        profile = "desktop_fullband"
        if rng.random() < 0.45:
            microphone = apply_random_bandwidth(microphone, rng)
    return microphone, profile


def valid_existing_audio(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        information = sf.info(path)
    except (OSError, RuntimeError):
        return False
    return (
        information.samplerate == SAMPLE_RATE
        and information.channels == 3
        and information.frames == SEQUENCE_SAMPLES
    )


def semantic_audio_is_valid(path: Path, row: dict[str, object]) -> bool:
    try:
        signals, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError):
        return False
    if sample_rate != SAMPLE_RATE or signals.shape != (SEQUENCE_SAMPLES, 3):
        return False
    microphone, render, target = signals.T
    mode = str(row["mode"])
    if active_rms(microphone) <= 1e-5:
        return False
    if mode in {"far_only", "mixture"} and active_rms(render) <= 1e-5:
        return False
    if mode in {"near_only", "mixture"} and active_rms(target) <= 1e-5:
        return False
    if mode == "far_only" and np.max(np.abs(target), initial=0.0) > 1e-5:
        return False
    if mode == "near_only" and np.max(np.abs(render), initial=0.0) > 1e-5:
        return False
    if str(row.get("reset_variant")) == "cold_onset":
        active = np.flatnonzero(labels_for(target) > 0.5)
        reset_frame = int(row["reset_frame"])
        if (
            not len(active)
            or int(active[0]) < reset_frame - 1
            or int(active[0]) > reset_frame + 19
        ):
            return False
    return True


def output_paths(root: Path, split: str, index: int) -> tuple[Path, Path]:
    shard = f"{index // 1000:03d}"
    audio = root / "materialized" / split / shard / f"{index:08d}.wav"
    metadata = root / "metadata" / split / shard / f"{index:08d}.json"
    return audio, metadata


def generate_example(
    target_root: Path,
    pools: SourcePools,
    split: str,
    index: int,
    config: V5MaterializationConfig,
) -> dict[str, object]:
    audio_path, metadata_path = output_paths(target_root, split, index)
    if valid_existing_audio(audio_path) and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("signal_validation_version") == 1
            and existing.get("reset_variant") != "cold_onset"
        ):
            return existing
        if semantic_audio_is_valid(audio_path, existing):
            existing["signal_validation_version"] = 1
            if existing.get("reset_variant") == "cold_onset":
                existing["cold_onset_alignment_version"] = 4
            atomic_write_json(metadata_path, existing)
            return existing

    example_seed = config.seed + (0 if split == "train" else 1_000_000_007)
    example_seed += index * 104_729
    rng = random.Random(example_seed)
    mode = choose_mode(rng, split)
    has_reset = rng.random() < config.reset_probability
    movement = mode != "near_only" and rng.random() < config.movement_probability
    noisy = rng.random() < config.noise_probability
    render_gap = mode != "near_only" and rng.random() < config.render_gap_probability
    use_real_echo = mode != "near_only" and rng.random() < config.real_echo_probability
    hard_far = mode == "far_only" and rng.random() < config.hard_far_probability

    if has_reset:
        reset_frame = rng.randint(20, 280)
        gap_hops = rng.randint(1, config.maximum_gap_hops)
        reset_variant = (
            "far_only"
            if mode == "far_only"
            else ("cold_onset" if rng.random() < 0.60 else "continuing")
        )
    else:
        reset_frame = 0
        gap_hops = 0
        reset_variant = "none"

    if mode == "far_only":
        propagated_nearend = np.zeros(SEQUENCE_SAMPLES, dtype=np.float32)
        near_metadata: dict[str, object] = {"type": "none", "group": "none"}
    else:
        propagated_nearend, near_metadata = choose_near_source(
            pools, rng, reset_frame, reset_variant
        )

    farend_nonlinear = False
    farend_noisy = False
    if mode == "near_only":
        echo = np.zeros(SEQUENCE_SAMPLES, dtype=np.float32)
        render = np.zeros(SEQUENCE_SAMPLES, dtype=np.float32)
        far_metadata: dict[str, object] = {"type": "none", "group": "none"}
        ser_db = 99.0
    else:
        for _ in range(8):
            (
                echo,
                render,
                far_metadata,
                farend_nonlinear,
                farend_noisy,
            ) = choose_far_source(
                pools,
                rng,
                movement=movement,
                use_real_echo=use_real_echo,
            )
            if active_rms(echo) > 1e-5 and active_rms(render) > 1e-5:
                break
        else:
            raise ValueError("could not select an active far-end/echo pair")
        if mode == "mixture":
            if rng.random() < 0.80:
                ser_db = rng.uniform(-18.0, 3.0)
            else:
                ser_db = rng.uniform(3.0, 12.0)
            desired_echo_rms = active_rms(propagated_nearend) / (
                10.0 ** (ser_db / 20.0)
            )
        else:
            ser_db = -99.0
            desired_echo_rms = 10.0 ** (rng.uniform(-28.0, -10.0) / 20.0)
        echo = scale_to_rms(echo, max(desired_echo_rms, 1e-6))

        if hard_far:
            echo = apply_smooth_gain_change(echo, rng)
            echo = apply_memory_nonlinearity(echo, rng)
            if rng.random() < 0.70:
                echo = apply_compressor(echo, rng)
            if rng.random() < 0.65:
                echo = apply_quantization(echo, rng.randint(6, 14))
            if rng.random() < 0.50:
                limit = rng.uniform(0.15, 0.85)
                echo = np.clip(echo, -limit, limit)
            render = apply_fractional_shift(render, rng.uniform(-1.0, 1.0))
            render *= 10.0 ** (rng.uniform(-18.0, 6.0) / 20.0)
            farend_nonlinear = True
        if render_gap:
            render = apply_reference_dropout(render, rng)

    microphone = propagated_nearend + echo
    noise_metadata: dict[str, object] = {"type": "none"}
    if noisy:
        noise_row = pools.noise[rng.randrange(len(pools.noise))]
        noise = crop_or_place(
            read_audio(Path(str(noise_row["path"]))),
            SEQUENCE_SAMPLES,
            rng,
            random_place=False,
        )
        snr_db = rng.uniform(3.0, 30.0)
        noise_rms = max(active_rms(microphone), 1e-4) / (10.0 ** (snr_db / 20.0))
        microphone += scale_to_rms(noise, noise_rms)
        noise_metadata = {
            "type": str(noise_row["corpus"]),
            "path": str(noise_row["path"]),
            "file_id": str(noise_row["file_id"]),
            "snr_db": snr_db,
        }

    microphone, device_profile = apply_device_profile(microphone, rng)
    render *= 10.0 ** (rng.uniform(-8.0, 8.0) / 20.0)
    microphone *= 10.0 ** (rng.uniform(-8.0, 8.0) / 20.0)
    if rng.random() < 0.15:
        limit = rng.uniform(0.20, 0.95)
        microphone = np.clip(microphone, -limit, limit)

    if gap_hops > 0:
        padding = gap_hops * HOP_LENGTH
        microphone[-padding:] = 0.0
        render[-padding:] = 0.0
        propagated_nearend[-padding:] = 0.0

    microphone = np.clip(sanitize(microphone), -1.0, 1.0)
    render = np.clip(sanitize(render), -1.0, 1.0)
    propagated_nearend = np.clip(sanitize(propagated_nearend), -1.0, 1.0)
    triplet = np.column_stack((microphone, render, propagated_nearend))

    scenario_parts = [mode, str(far_metadata["type"]), device_profile]
    if movement:
        scenario_parts.append("movement")
    if noisy:
        scenario_parts.append("noisy")
    if hard_far:
        scenario_parts.append("hard_far")
    if render_gap:
        scenario_parts.append("render_gap")
    if has_reset:
        scenario_parts.append("reset_" + reset_variant)
    scenario = "+".join(scenario_parts)
    cluster_group = (
        str(far_metadata["group"])
        if mode == "far_only"
        else str(near_metadata["group"])
    )
    row: dict[str, object] = {
        "audio": str(audio_path.relative_to(target_root)),
        "file_id": f"v5-{split}-{index:08d}",
        "cluster_id": (
            f"far:{far_metadata['type']}:{cluster_group}"
            if mode == "far_only"
            else f"near:{near_metadata['type']}:{cluster_group}"
        ),
        "mode": mode,
        "scenario": scenario,
        "ser_db": ser_db,
        "farend_nonlinear": farend_nonlinear,
        "farend_noisy": farend_noisy,
        "nearend_noisy": noisy and mode != "far_only",
        "movement": movement,
        "render_gap": render_gap,
        "hard_far": hard_far,
        "device_profile": device_profile,
        "reset_frame": reset_frame,
        "gap_hops": gap_hops,
        "reset_variant": reset_variant,
        "cold_onset_alignment_version": 4,
        "signal_validation_version": 1,
        "example_seed": example_seed,
        "sources": {
            "near": near_metadata,
            "far": far_metadata,
            "noise": noise_metadata,
        },
    }

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_audio = audio_path.with_suffix(".writing.wav")
    sf.write(
        temporary_audio,
        triplet,
        SAMPLE_RATE,
        subtype="PCM_16",
        format="WAV",
    )
    os.replace(temporary_audio, audio_path)
    atomic_write_json(metadata_path, row)
    return row


def materialize_split(
    target_root: Path,
    split: str,
    examples: int,
    workers: int,
    config: V5MaterializationConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    pools = load_source_pools(target_root, split)
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    def run(index: int) -> dict[str, object]:
        try:
            return generate_example(target_root, pools, split, index, config)
        except Exception as error:  # The failure record is required for resume.
            return {
                "_failure": True,
                "split": split,
                "index": index,
                "error_type": type(error).__name__,
                "error": str(error),
            }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for completed, result in enumerate(executor.map(run, range(examples)), start=1):
            if result.pop("_failure", False):
                failures.append(result)
            else:
                rows.append(result)
            if completed % 500 == 0 or completed == examples:
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "split": split,
                            "completed": completed,
                            "total": examples,
                            "failures": len(failures),
                            "elapsed_seconds": round(elapsed, 2),
                            "examples_per_second": round(completed / elapsed, 3),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    rows.sort(key=lambda row: str(row["file_id"]))
    failures.sort(key=lambda row: int(row["index"]))
    return rows, failures, time.perf_counter() - started


def manifest_statistics(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "examples": len(rows),
        "mode_counts": dict(Counter(str(row["mode"]) for row in rows)),
        "near_source_counts": dict(
            Counter(str(row["sources"]["near"]["type"]) for row in rows)  # type: ignore[index]
        ),
        "far_source_counts": dict(
            Counter(str(row["sources"]["far"]["type"]) for row in rows)  # type: ignore[index]
        ),
        "device_profile_counts": dict(
            Counter(str(row["device_profile"]) for row in rows)
        ),
        "movement": sum(bool(row["movement"]) for row in rows),
        "render_gap": sum(bool(row["render_gap"]) for row in rows),
        "hard_far": sum(bool(row["hard_far"]) for row in rows),
        "reset": sum(int(row["reset_frame"]) > 0 for row in rows),
        "reset_variant_counts": dict(
            Counter(str(row["reset_variant"]) for row in rows)
        ),
    }


def assert_no_but_leakage(
    manifests: dict[str, list[dict[str, object]]],
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for field in ("utterance_id", "speaker", "room"):
        values = {
            split: {str(row[field]) for row in manifests[split]}
            for split in ("train", "validation", "test")
        }
        checks[field] = not (
            values["train"] & values["validation"]
            or values["train"] & values["test"]
            or values["validation"] & values["test"]
        )
    if not all(checks.values()):
        raise ValueError(f"BUT replay split leakage detected: {checks}")
    return checks


def write_dataset_card(target_root: Path, summary: dict[str, object]) -> None:
    materialized = summary["materialized"]  # type: ignore[index]
    text = f"""# Render-aware DTD v5 dataset

This dataset preserves every v4 manifest byte-for-byte and adds precomputed
8-second, 16 kHz, three-channel PCM16 examples. Channels are microphone,
render, and propagated near-end target. Use `train_materialized_v5.py` to
select these manifests without changing the frozen candidate-16 loader.

## Materialized counts

- train: {materialized['train']['examples']}
- validation: {materialized['validation']['examples']}
- seed: {summary['configuration']['seed']}
- sequence: 8 seconds

## Leakage controls

BUT retransmitted LibriSpeech is split by both source speaker and room. An
utterance, speaker, or room cannot cross train/validation/test. Existing v4
assignments are unchanged. VoiceBox benchmark audio is not referenced.

## Explicit exclusions

The following wake-word/hotword corpora are not discovered or referenced:

{os.linesep.join('- ' + name for name in EXCLUDED_KWS_CORPORA)}

## Provenance

BUT ReverbDB is CC BY 4.0 and requires attribution. See the source corpus
`read_me.txt`. Full source paths, seeds, scenario flags, and generation
parameters are retained in the manifests and per-example metadata.
"""
    (target_root / "DATASET_CARD.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    v4_root = Path(args.v4_root).resolve()
    but_root = Path(args.but_replayed_root).resolve()
    target_root = Path(args.target_root).resolve()
    if target_root == v4_root or v4_root in target_root.parents:
        raise ValueError("v5 target must not be v4 or a child of v4")
    if not v4_root.is_dir() or not but_root.is_dir():
        raise FileNotFoundError("v4 or BUT retransmission source directory is missing")
    if target_root.exists() and any(target_root.iterdir()) and not args.resume:
        raise FileExistsError(
            f"target is not empty; pass --resume to continue safely: {target_root}"
        )
    if args.train_examples <= 0 or args.validation_examples <= 0:
        raise ValueError("materialized example counts must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    target_root.mkdir(parents=True, exist_ok=True)

    config = V5MaterializationConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        validation_examples=args.validation_examples,
    )
    atomic_write_json(target_root / "generation_config.json", asdict(config))
    base_manifest_hashes = copy_v4_manifests(v4_root, target_root)
    previous_summary_path = target_root / "dataset_summary.json"
    cached_but_manifests = all(
        (target_root / "manifests" / f"but_replayed_{split}.jsonl").is_file()
        for split in ("train", "validation", "test")
    )
    if args.resume and previous_summary_path.is_file() and cached_but_manifests:
        previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8"))
        but_summary = previous_summary["but_replayed"]
        leakage_checks = previous_summary["leakage_checks"]
    else:
        but_manifests, but_summary = discover_but_replayed(but_root, args.seed)
        leakage_checks = assert_no_but_leakage(but_manifests)
        for split in ("train", "validation", "test"):
            atomic_write_jsonl(
                target_root / "manifests" / f"but_replayed_{split}.jsonl",
                but_manifests[split],
            )

    all_failures: list[dict[str, object]] = []
    durations: dict[str, float] = {}
    materialized_rows: dict[str, list[dict[str, object]]] = {}
    for split, examples in (
        ("train", args.train_examples),
        ("validation", args.validation_examples),
    ):
        rows, failures, duration = materialize_split(
            target_root, split, examples, args.workers, config
        )
        all_failures.extend(failures)
        durations[split] = duration
        materialized_rows[split] = rows
        atomic_write_jsonl(
            target_root / "manifests" / f"materialized_{split}.jsonl", rows
        )
        if failures:
            atomic_write_jsonl(target_root / "generation_failures.jsonl", all_failures)
            raise RuntimeError(
                f"{len(failures)} {split} examples failed; rerun with --resume"
            )

    audio_files = list((target_root / "materialized").rglob("*.wav"))
    v4_summary = json.loads(
        (v4_root / "dataset_summary.json").read_text(encoding="utf-8")
    )
    summary: dict[str, Any] = {
        "schema_version": "3.0-materialized",
        "created_utc_unix": int(time.time()),
        "roots": {
            "v4": str(v4_root),
            "but_replayed": str(but_root),
            "target": str(target_root),
        },
        "configuration": asdict(config),
        "base_v4_manifest_hashes": base_manifest_hashes,
        "base_v4_manifest_counts": v4_summary["manifest_counts"],
        "but_replayed": but_summary,
        "leakage_checks": leakage_checks,
        "materialized": {
            split: manifest_statistics(materialized_rows[split])
            for split in ("train", "validation")
        },
        "materialized_audio": {
            "files": len(audio_files),
            "bytes": sum(path.stat().st_size for path in audio_files),
            "sample_rate": SAMPLE_RATE,
            "channels": ["microphone", "render", "propagated_nearend"],
            "subtype": "PCM_16",
            "sequence_seconds": SEQUENCE_SECONDS,
        },
        "generation_seconds": durations,
        "failures": all_failures,
        "excluded_kws_corpora": list(EXCLUDED_KWS_CORPORA),
    }
    atomic_write_json(target_root / "dataset_summary.json", summary)
    write_dataset_card(target_root, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
