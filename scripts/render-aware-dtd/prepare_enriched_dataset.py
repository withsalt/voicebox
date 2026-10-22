from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

from audio_io import AudioMetadata, inspect_audio


PATH_FIELDS = ("path", "microphone", "render", "nearend", "target", "echo")
MANIFEST_NAMES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic render-aware DTD manifests without copying "
            "the source corpora. Every selected audio header is validated."
        )
    )
    parser.add_argument(
        "--aec-root",
        default=r"E:\Projects\render-aware-dtd\datasets\AEC-Challenge",
    )
    parser.add_argument(
        "--wenet-root",
        default=r"E:\Projects\render-aware-dtd\datasets\WenetSpeech4TTS",
    )
    parser.add_argument(
        "--rir-root",
        default=(
            r"E:\Projects\render-aware-dtd\datasets"
            r"\BUT_ReverbDB_rel_19_06_RIR-Only"
        ),
    )
    parser.add_argument(
        "--near-root",
        default=r"E:\Projects\kws-train\datasets",
    )
    parser.add_argument(
        "--target-root",
        default=r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--skip-header-validation",
        action="store_true",
        help="Intended only for manifest-builder development smoke tests.",
    )
    return parser.parse_args()


def stable_bucket(value: str, buckets: int = 100) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % buckets


def hashed_split(group: str) -> str:
    bucket = stable_bucket(group)
    if bucket < 5:
        return "test"
    if bucket < 10:
        return "validation"
    return "train"


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    os.replace(temporary, path)
    return count


def require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {resolved}")
    return resolved


def discover_aec_stems(
    aec_root: Path,
    manifests: dict[str, list[dict[str, object]]],
) -> None:
    narrow_root = aec_root / "datasets" / "synthetic"
    with (narrow_root / "meta.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for source in csv.DictReader(stream):
            file_id = int(source["fileid"])
            split = (
                "test"
                if source["split"].strip().lower() == "test"
                else (
                    "validation"
                    if stable_bucket(f"narrow:{file_id}") < 5
                    else "train"
                )
            )
            manifests[f"narrow_{split}"].append(
                {
                    "file_id": str(file_id),
                    "microphone": str(
                        (
                            narrow_root
                            / "nearend_mic_signal"
                            / f"nearend_mic_fileid_{file_id}.wav"
                        ).resolve()
                    ),
                    "render": str(
                        (
                            narrow_root
                            / "farend_speech"
                            / f"farend_speech_fileid_{file_id}.wav"
                        ).resolve()
                    ),
                    "nearend": str(
                        (
                            narrow_root
                            / "nearend_speech"
                            / f"nearend_speech_fileid_{file_id}.wav"
                        ).resolve()
                    ),
                    "target": str(
                        (
                            narrow_root
                            / "nearend_speech"
                            / f"nearend_speech_fileid_{file_id}.wav"
                        ).resolve()
                    ),
                    "echo": str(
                        (
                            narrow_root
                            / "echo_signal"
                            / f"echo_fileid_{file_id}.wav"
                        ).resolve()
                    ),
                    "nearend_scale": float(source["nearend_scale"]),
                    "ser_db": float(source["ser"]),
                    "farend_nonlinear": bool(
                        int(source["is_farend_nonlinear"])
                    ),
                    "farend_noisy": bool(int(source["is_farend_noisy"])),
                    "nearend_noisy": bool(int(source["is_nearend_noisy"])),
                }
            )

    fullband_root = aec_root / "datasets" / "synthetic_fullband"
    with (fullband_root / "meta.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for source in csv.DictReader(stream):
            file_id = source["file_id"]
            split = hashed_split(f"fullband:{file_id}")
            base = fullband_root / "data" / file_id
            manifests[f"fullband_{split}"].append(
                {
                    "file_id": file_id,
                    "microphone": str(
                        base.with_name(base.name + "_mic.wav").resolve()
                    ),
                    "render": str(
                        base.with_name(base.name + "_farend.wav").resolve()
                    ),
                    "nearend": str(
                        base.with_name(base.name + "_nearend.wav").resolve()
                    ),
                    "target": str(
                        base.with_name(base.name + "_target.wav").resolve()
                    ),
                    "echo": str(
                        base.with_name(base.name + "_echo.wav").resolve()
                    ),
                    "nearend_scale": 1.0,
                    "ser_db": float(source["ser"]),
                    "farend_nonlinear": False,
                    "farend_noisy": source["is_farend_noisy"].lower() == "true",
                    "nearend_noisy": source["is_nearend_noisy"].lower() == "true",
                }
            )


def discover_wenet(
    wenet_root: Path,
    manifests: dict[str, list[dict[str, object]]],
) -> None:
    subsets = (
        ("Premium", wenet_root / "Premium"),
        ("Standard", wenet_root / "Standard"),
    )
    discovered = 0
    for quality, root in subsets:
        for path in sorted(root.glob("*/wavs/*.wav")):
            source_group = path.stem.split("_", maxsplit=1)[0]
            split = hashed_split(f"wenet:{source_group}")
            manifests[f"farend_{split}"].append(
                {
                    "path": str(path.resolve()),
                    "file_id": path.stem,
                    "group": source_group,
                    "corpus": f"wenet_{quality.lower()}",
                }
            )
            discovered += 1
    if discovered == 0:
        raise FileNotFoundError(f"no WenetSpeech4TTS WAV files found: {wenet_root}")


def iter_wavs(path: Path) -> Iterator[Path]:
    if not path.is_dir():
        return
    yield from sorted(path.rglob("*.wav"))


def append_near_files(
    manifests: dict[str, list[dict[str, object]]],
    paths: Iterable[Path],
    *,
    split: str,
    corpus: str,
    group_from_parent: bool = True,
) -> None:
    for path in paths:
        group = path.parent.name if group_from_parent else path.stem.split("_")[0]
        manifests[f"nearend_{split}"].append(
            {
                "path": str(path.resolve()),
                "file_id": path.stem,
                "group": group,
                "corpus": corpus,
            }
        )


def discover_nearend(
    near_root: Path,
    manifests: dict[str, list[dict[str, object]]],
) -> None:
    aishell = near_root / "speech_asr_aishell1_trainsets"
    for split, source_name in (
        ("train", "train"),
        ("validation", "dev"),
        ("test", "test"),
    ):
        source = (
            aishell
            / f"speech_asr_aishell_{source_name}sets"
            / "wav"
            / source_name
        )
        append_near_files(
            manifests,
            iter_wavs(source),
            split=split,
            corpus="aishell1",
        )

    aishell3_train = near_root / "aishell3_16k" / "train" / "wav"
    for speaker_directory in sorted(
        path for path in aishell3_train.iterdir() if path.is_dir()
    ):
        split = hashed_split(f"aishell3:{speaker_directory.name}")
        append_near_files(
            manifests,
            iter_wavs(speaker_directory),
            split=split,
            corpus="aishell3",
        )

    thchs = near_root / "THCHS30_2015" / "data_thchs30"
    for split, source_name in (
        ("train", "train"),
        ("validation", "dev"),
        ("test", "test"),
    ):
        append_near_files(
            manifests,
            iter_wavs(thchs / source_name),
            split=split,
            corpus="thchs30",
            group_from_parent=False,
        )

    if not manifests["nearend_train"]:
        raise FileNotFoundError(f"no near-end speech WAV files found: {near_root}")


def discover_noise(
    near_root: Path,
    rir_root: Path,
    manifests: dict[str, list[dict[str, object]]],
) -> None:
    sources = (
        (
            "dns_noise",
            near_root
            / "ICASSP_2021_DNS_Challenge"
            / "datasets"
            / "datasets"
            / "noise",
        ),
        ("musan_noise", near_root / "musan" / "noise"),
        ("musan_music", near_root / "musan" / "music"),
    )
    for corpus, root in sources:
        for path in iter_wavs(root):
            split = hashed_split(f"{corpus}:{path.stem}")
            manifests[f"noise_{split}"].append(
                {
                    "path": str(path.resolve()),
                    "file_id": path.stem,
                    "group": path.parent.name,
                    "corpus": corpus,
                }
            )

    for path in iter_wavs(rir_root):
        if "silence" not in {part.lower() for part in path.parts}:
            continue
        room = path.relative_to(rir_root).parts[0]
        split = room_split(room)
        manifests[f"noise_{split}"].append(
            {
                "path": str(path.resolve()),
                "file_id": path.stem,
                "group": room,
                "corpus": "but_room_silence",
            }
        )
    if not manifests["noise_train"]:
        raise FileNotFoundError("no training noise WAV files found")


def room_split(room: str) -> str:
    if room == "VUT_FIT_L227":
        return "test"
    if room == "Hotel_SkalskyDvur_Room112":
        return "validation"
    return "train"


def discover_rirs(
    rir_root: Path,
    manifests: dict[str, list[dict[str, object]]],
) -> None:
    for path in iter_wavs(rir_root):
        if "rir" not in {part.lower() for part in path.parts}:
            continue
        relative = path.relative_to(rir_root)
        room = relative.parts[0]
        split = room_split(room)
        manifests[f"rir_{split}"].append(
            {
                "path": str(path.resolve()),
                "file_id": str(relative.with_suffix("")),
                "group": room,
                "corpus": "but_reverbdb",
            }
        )
    for split in ("train", "validation", "test"):
        if not manifests[f"rir_{split}"]:
            raise FileNotFoundError(f"no {split} RIR WAV files found: {rir_root}")


def real_recording_id(name: str, marker: str) -> str:
    index = name.find(marker)
    if index <= 0:
        raise ValueError(f"cannot parse AEC real recording name: {name}")
    return name[:index]


def discover_real_aec(
    aec_root: Path,
    manifests: dict[str, list[dict[str, object]]],
) -> None:
    datasets_root = aec_root / "datasets"
    collections = (
        "real",
        "real_doubled",
        "real_doubled_fullband",
        "real_doubled_fullband_mobile",
    )
    missing_pairs: list[str] = []
    for collection in collections:
        root = datasets_root / collection
        if not root.is_dir():
            continue
        for render in sorted(root.glob("*_farend_singletalk*_lpb.wav")):
            microphone = render.with_name(
                render.name.removesuffix("_lpb.wav") + "_mic.wav"
            )
            if not microphone.is_file():
                missing_pairs.append(str(microphone))
                continue
            marker = "_farend_singletalk"
            recording_id = real_recording_id(render.name, marker)
            split = hashed_split(f"aec-real:{collection}:{recording_id}")
            manifests[f"real_{split}"].append(
                {
                    "microphone": str(microphone.resolve()),
                    "render": str(render.resolve()),
                    "mode": "far_only",
                    "collection": collection,
                    "recording_id": recording_id + render.stem[len(recording_id) :],
                }
            )
        for microphone in sorted(root.glob("*_nearend_singletalk_mic.wav")):
            marker = "_nearend_singletalk"
            recording_id = real_recording_id(microphone.name, marker)
            split = hashed_split(f"aec-real:{collection}:{recording_id}")
            manifests[f"real_{split}"].append(
                {
                    "microphone": str(microphone.resolve()),
                    "render": "",
                    "mode": "near_only",
                    "collection": collection,
                    "recording_id": recording_id + "_nearend",
                }
            )
    if missing_pairs:
        print(
            json.dumps(
                {
                    "warning": "skipped AEC real loopbacks without microphone pairs",
                    "missing_pairs": len(missing_pairs),
                    "first_missing": missing_pairs[:5],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def referenced_paths(
    manifests: dict[str, list[dict[str, object]]],
) -> list[str]:
    paths: set[str] = set()
    for rows in manifests.values():
        for row in rows:
            for field in PATH_FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value:
                    paths.add(value)
    return sorted(paths)


def inspect_one(
    path: str,
) -> tuple[AudioMetadata | None, tuple[str, str] | None]:
    try:
        return inspect_audio(path), None
    except (OSError, RuntimeError, ValueError) as error:
        return None, (path, str(error))


def inspect_all(
    paths: list[str],
    workers: int,
) -> tuple[dict[str, AudioMetadata], dict[str, str]]:
    metadata: dict[str, AudioMetadata] = {}
    invalid: dict[str, str] = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, (item, failure) in enumerate(
            executor.map(inspect_one, paths, chunksize=64),
            start=1,
        ):
            if item is not None:
                metadata[item.path] = item
            elif failure is not None:
                invalid[str(Path(failure[0]).resolve())] = failure[1]
            if index % 25_000 == 0 or index == len(paths):
                print(
                    json.dumps(
                        {
                            "validated_audio_files": index,
                            "total_audio_files": len(paths),
                            "seconds": round(time.perf_counter() - started, 1),
                        }
                    ),
                    flush=True,
                )
    return metadata, invalid


def drop_invalid_rows(
    manifests: dict[str, list[dict[str, object]]],
    invalid: dict[str, str],
) -> dict[str, int]:
    dropped: dict[str, int] = {}
    if not invalid:
        return dropped
    invalid_paths = set(invalid)
    for name, rows in manifests.items():
        retained: list[dict[str, object]] = []
        for row in rows:
            references = {
                str(Path(value).resolve())
                for field in PATH_FIELDS
                if isinstance((value := row.get(field)), str) and value
            }
            if references & invalid_paths:
                continue
            retained.append(row)
        if len(retained) != len(rows):
            dropped[name] = len(rows) - len(retained)
            manifests[name] = retained
    return dropped


def attach_audio_metadata(
    manifests: dict[str, list[dict[str, object]]],
    metadata: dict[str, AudioMetadata],
) -> None:
    if not metadata:
        return
    for rows in manifests.values():
        for row in rows:
            source_formats: dict[str, dict[str, int | float]] = {}
            for field in PATH_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value:
                    continue
                item = metadata[str(Path(value).resolve())]
                source_formats[field] = {
                    "sample_rate": item.sample_rate,
                    "channels": item.channels,
                    "frames": item.frames,
                    "duration_seconds": round(item.duration_seconds, 6),
                }
            row["source_formats"] = source_formats


def summarize(
    manifests: dict[str, list[dict[str, object]]],
    metadata: dict[str, AudioMetadata],
    roots: dict[str, Path],
    invalid: dict[str, str],
    dropped_rows: dict[str, int],
) -> dict[str, object]:
    sample_rates = Counter(item.sample_rate for item in metadata.values())
    channel_counts = Counter(item.channels for item in metadata.values())
    corpus_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for name, rows in manifests.items():
        for row in rows:
            corpus_counts[name][str(row.get("corpus", "aec"))] += 1
    return {
        "schema_version": "2.0",
        "created_utc_unix": int(time.time()),
        "roots": {name: str(path) for name, path in roots.items()},
        "manifest_counts": {
            name: len(manifests[name]) for name in MANIFEST_NAMES
        },
        "corpus_counts": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(corpus_counts.items())
        },
        "validated_audio_files": len(metadata),
        "invalid_audio_files": len(invalid),
        "invalid_audio_first": [
            {"path": path, "error": error}
            for path, error in list(sorted(invalid.items()))[:20]
        ],
        "dropped_invalid_rows": dict(sorted(dropped_rows.items())),
        "source_sample_rates": {
            str(rate): count for rate, count in sorted(sample_rates.items())
        },
        "source_channel_counts": {
            str(channels): count
            for channels, count in sorted(channel_counts.items())
        },
        "audio_contract": {
            "runtime_header_read": True,
            "target_sample_rate": 16_000,
            "target_channels": 1,
            "resampler": "scipy.signal.resample_poly",
        },
        "splitting": {
            "hash": "sha256",
            "generic": "90% train / 5% validation / 5% test by group",
            "aec_narrow_test": "official test split",
            "but_rir_validation_room": "Hotel_SkalskyDvur_Room112",
            "but_rir_test_room": "VUT_FIT_L227",
        },
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    roots = {
        "aec": require_directory(Path(args.aec_root), "AEC Challenge"),
        "wenet": require_directory(Path(args.wenet_root), "WenetSpeech4TTS"),
        "rir": require_directory(Path(args.rir_root), "BUT ReverbDB RIR"),
        "near": require_directory(Path(args.near_root), "near-end/noise"),
    }
    target_root = Path(args.target_root).resolve()
    manifests: dict[str, list[dict[str, object]]] = {
        name: [] for name in MANIFEST_NAMES
    }

    print('{"stage":"discover_aec_stems"}', flush=True)
    discover_aec_stems(roots["aec"], manifests)
    print('{"stage":"discover_wenet_farend"}', flush=True)
    discover_wenet(roots["wenet"], manifests)
    print('{"stage":"discover_nearend"}', flush=True)
    discover_nearend(roots["near"], manifests)
    print('{"stage":"discover_noise"}', flush=True)
    discover_noise(roots["near"], roots["rir"], manifests)
    print('{"stage":"discover_rirs"}', flush=True)
    discover_rirs(roots["rir"], manifests)
    print('{"stage":"discover_real_aec"}', flush=True)
    discover_real_aec(roots["aec"], manifests)
    for name in MANIFEST_NAMES:
        if not manifests[name]:
            raise ValueError(f"manifest would be empty: {name}")

    paths = referenced_paths(manifests)
    print(
        json.dumps(
            {
                "stage": "validate_headers",
                "referenced_audio_files": len(paths),
                "skip": args.skip_header_validation,
            }
        ),
        flush=True,
    )
    metadata, invalid = (
        ({}, {})
        if args.skip_header_validation
        else inspect_all(paths, args.workers)
    )
    dropped_rows = drop_invalid_rows(manifests, invalid)
    for name in MANIFEST_NAMES:
        if not manifests[name]:
            raise ValueError(
                f"manifest is empty after invalid-audio filtering: {name}"
            )
    attach_audio_metadata(manifests, metadata)

    manifest_root = target_root / "manifests"
    counts = {
        name: atomic_write_jsonl(
            manifest_root / f"{name}.jsonl",
            manifests[name],
        )
        for name in MANIFEST_NAMES
    }
    summary = summarize(
        manifests,
        metadata,
        roots,
        invalid,
        dropped_rows,
    )
    summary["manifest_counts"] = counts
    target_root.mkdir(parents=True, exist_ok=True)
    summary_path = target_root / "dataset_summary.json"
    temporary = summary_path.with_suffix(".json.writing")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
