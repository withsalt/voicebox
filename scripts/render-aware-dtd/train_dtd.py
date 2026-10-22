from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import pickle
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aec_dataset import AecChallengeDtdDataset
from dtd_event_metrics import event_metrics
from dtd_metrics import binary_metrics, low_fpr_operating_points, select_threshold
from dtd_model import DtdModelConfig, RenderAwareDtd
from dtd_training_objectives import (
    EventObjectiveConfig,
    apply_stream_resets,
    training_losses,
)
from enriched_dataset import create_enriched_datasets

CHECKPOINT_FORMAT_VERSION = 4
SELECTION_POLICY_VERSION = "low_fpr_grid_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a streaming render-aware near-end/double-talk detector."
    )
    parser.add_argument(
        "--dataset-root",
        default=r"E:\Projects\render-aware-dtd\datasets\AEC-Challenge",
    )
    parser.add_argument(
        "--enriched-root",
        default=r"E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v5",
        help=(
            "Self-contained enriched dataset root. When supplied, this "
            "replaces the legacy AEC-only training dataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "artifacts"),
    )
    parser.add_argument("--epochs", type=int, default=230)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-seconds", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--positive-weight", type=float, default=1.5)
    parser.add_argument("--hard-negative-loss-weight", type=float, default=0.15)
    parser.add_argument("--onset-loss-weight", type=float, default=0.10)
    parser.add_argument("--event-loss-weight", type=float, default=0.0)
    parser.add_argument("--reset-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--hard-negative-top-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--event-decision-threshold", type=float, default=0.55)
    parser.add_argument("--event-temperature", type=float, default=0.08)
    parser.add_argument(
        "--model-size",
        type=int,
        choices=(192, 256, 288),
        default=288,
        help="Projection and GRU hidden size.",
    )
    parser.add_argument(
        "--gru-layers",
        type=int,
        choices=(2, 3),
        default=3,
        help="Number of recurrent GRU layers.",
    )
    parser.add_argument(
        "--frequency-bands",
        type=int,
        choices=(48, 64, 80),
        default=80,
        help=(
            "Fixed mel filterbank band count. Only the in-graph front end "
            "changes; the recurrent state contract is unaffected."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--aec-narrow-weight",
        type=float,
        default=0.40,
        help="Sampling weight for AEC narrowband stems.",
    )
    parser.add_argument(
        "--aec-fullband-weight",
        type=float,
        default=0.20,
        help="Sampling weight for AEC fullband stems.",
    )
    parser.add_argument(
        "--wenet-weight",
        type=float,
        default=0.20,
        help="Sampling weight for Wenet online synthetic mixtures.",
    )
    parser.add_argument(
        "--wenet-far-only-probability",
        type=float,
        default=0.40,
        help=(
            "Far-end-only probability for online Wenet training synthesis. "
            "Validation and test distributions remain fixed."
        ),
    )
    parser.add_argument("--v5-reset-probability", type=float, default=0.2)
    parser.add_argument(
        "--v5-hard-far-only-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--v5-maximum-gap-hops", type=int, default=8)
    parser.add_argument(
        "--v5-validation-reset-probability",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--v5-validation-hard-far-only-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--v5-validation-maximum-gap-hops",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--aec-real-weight",
        type=float,
        default=0.20,
        help="Sampling weight for real AEC single-talk recordings.",
    )
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--validation-modulo", type=int, default=20)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument(
        "--examples-per-epoch",
        type=int,
        default=12_000,
        help="Fixed mixed-sample budget used with --enriched-root.",
    )
    parser.add_argument(
        "--validation-examples",
        type=int,
        default=1_500,
        help="Fixed Wenet-based validation scenario count.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help=(
            "Stop after this many epochs without a material validation AP "
            "improvement. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=2e-4,
        help="Minimum validation AP increase that resets early-stopping patience.",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    # This environment's cuDNN convolution/RNN sublibraries are different
    # patch versions and can terminate Python after a successful epoch.
    # Native CUDA kernels remain accelerated and exit cleanly.
    torch.backends.cudnn.enabled = False
    probe = torch.zeros(1, 1, 640, device=device)
    kernel = torch.zeros(2, 1, 32, device=device)
    F.conv1d(probe, kernel)
    torch.cuda.synchronize(device)


def create_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # Workers are recreated each epoch so set_epoch() changes augmentations.
        persistent_workers=False,
        drop_last=shuffle,
        generator=generator,
    )


@dataclass
class ResourceMonitor:
    process: psutil.Process
    peak_process_rss_bytes: int = 0
    peak_process_tree_rss_bytes: int = 0
    peak_process_working_set_bytes: int = 0
    peak_system_used_bytes: int = 0
    minimum_system_available_bytes: int = 2**63 - 1
    peak_commit_used_bytes: int = 0
    commit_limit_bytes: int = 0
    peak_commit_fraction: float = 0.0
    commit_limit_at_peak_fraction_bytes: int = 0

    @classmethod
    def create(cls) -> "ResourceMonitor":
        monitor = cls(process=psutil.Process())
        monitor.sample()
        return monitor

    def sample(self) -> None:
        memory = self.process.memory_info()
        self.peak_process_rss_bytes = max(
            self.peak_process_rss_bytes,
            int(memory.rss),
        )
        process_tree_rss = int(memory.rss)
        for child in self.process.children(recursive=True):
            try:
                process_tree_rss += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_process_tree_rss_bytes = max(
            self.peak_process_tree_rss_bytes,
            process_tree_rss,
        )
        self.peak_process_working_set_bytes = max(
            self.peak_process_working_set_bytes,
            int(getattr(memory, "peak_wset", memory.rss)),
        )
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        system_used = int(virtual.total - virtual.available)
        if os.name == "nt":
            commit_used, commit_limit = windows_commit_memory()
        else:
            commit_used = system_used + int(swap.used)
            commit_limit = int(virtual.total + swap.total)
        self.peak_system_used_bytes = max(
            self.peak_system_used_bytes,
            system_used,
        )
        self.minimum_system_available_bytes = min(
            self.minimum_system_available_bytes,
            int(virtual.available),
        )
        self.peak_commit_used_bytes = max(
            self.peak_commit_used_bytes,
            commit_used,
        )
        self.commit_limit_bytes = max(self.commit_limit_bytes, commit_limit)
        commit_fraction = commit_used / max(1, commit_limit)
        if commit_fraction > self.peak_commit_fraction:
            self.peak_commit_fraction = commit_fraction
            self.commit_limit_at_peak_fraction_bytes = commit_limit

    def to_dict(self, device: torch.device) -> dict[str, int | float]:
        self.sample()
        values: dict[str, int | float] = {
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
            "peak_process_tree_rss_bytes": self.peak_process_tree_rss_bytes,
            "peak_process_working_set_bytes": (self.peak_process_working_set_bytes),
            "peak_system_used_bytes": self.peak_system_used_bytes,
            "minimum_system_available_bytes": (self.minimum_system_available_bytes),
            "peak_commit_used_bytes": self.peak_commit_used_bytes,
            "commit_limit_bytes": self.commit_limit_bytes,
            "peak_commit_fraction": self.peak_commit_fraction,
            "commit_limit_at_peak_fraction_bytes": (
                self.commit_limit_at_peak_fraction_bytes
            ),
        }
        if device.type == "cuda":
            values.update(
                {
                    "peak_cuda_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_cuda_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device)
                    ),
                }
            )
        return values


def windows_commit_memory() -> tuple[int, int]:
    if os.name != "nt":
        raise OSError("Windows commit accounting is only available on Windows")

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("commit_total", ctypes.c_size_t),
            ("commit_limit", ctypes.c_size_t),
            ("commit_peak", ctypes.c_size_t),
            ("physical_total", ctypes.c_size_t),
            ("physical_available", ctypes.c_size_t),
            ("system_cache", ctypes.c_size_t),
            ("kernel_total", ctypes.c_size_t),
            ("kernel_paged", ctypes.c_size_t),
            ("kernel_nonpaged", ctypes.c_size_t),
            ("page_size", ctypes.c_size_t),
            ("handle_count", ctypes.c_ulong),
            ("process_count", ctypes.c_ulong),
            ("thread_count", ctypes.c_ulong),
        ]

    information = PerformanceInformation()
    information.size = ctypes.sizeof(PerformanceInformation)
    get_performance_info = ctypes.windll.psapi.GetPerformanceInfo
    if not get_performance_info(
        ctypes.byref(information),
        information.size,
    ):
        raise ctypes.WinError()
    return (
        int(information.commit_total * information.page_size),
        int(information.commit_limit * information.page_size),
    )


def learning_rate_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if args.lr_schedule == "constant":
        return args.learning_rate
    if epoch < args.warmup_epochs:
        return args.learning_rate * (epoch + 1) / max(1, args.warmup_epochs)
    decay_epochs = max(1, args.epochs - args.warmup_epochs - 1)
    progress = min(
        1.0,
        max(0.0, (epoch - args.warmup_epochs) / decay_epochs),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        args.min_learning_rate + (args.learning_rate - args.min_learning_rate) * cosine
    )


def run_training_epoch(
    model: RenderAwareDtd,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    positive_weight: float,
    objective_config: EventObjectiveConfig,
    resource_monitor: ResourceMonitor,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    component_totals = {
        "bce": 0.0,
        "hard_negative": 0.0,
        "onset": 0.0,
        "event": 0.0,
        "reset": 0.0,
    }
    total_gradient_norm = 0.0
    optimizer_steps = 0
    total_frames = 0
    true_positive = false_positive = false_negative = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        microphone = batch["microphone"].to(device, non_blocking=True)
        render = batch["render"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid_mask = batch.get("valid_mask")
        valid_mask = (
            torch.ones_like(target, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.to(device, non_blocking=True, dtype=torch.bool)
        )
        reset_before = batch.get("reset_before")
        reset_before = (
            torch.zeros_like(target, dtype=torch.bool)
            if reset_before is None
            else reset_before.to(device, non_blocking=True, dtype=torch.bool)
        )
        modes = batch.get("mode", ["mixture"] * target.shape[0])
        far_only_mask = torch.tensor(
            [mode == "far_only" for mode in modes],
            device=device,
            dtype=torch.bool,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits, _ = model(microphone, render)
            logits = apply_stream_resets(
                model,
                logits,
                microphone,
                render,
                reset_before,
                model.config.hop_length,
            )
            components = training_losses(
                logits,
                target,
                valid_mask,
                reset_before,
                far_only_mask,
                positive_weight,
                objective_config,
            )
            loss = components["total"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        resource_monitor.sample()

        with torch.no_grad():
            predicted = logits >= 0.0
            actual = target >= 0.5
            true_positive += int(torch.count_nonzero(predicted & actual & valid_mask))
            false_positive += int(torch.count_nonzero(predicted & ~actual & valid_mask))
            false_negative += int(torch.count_nonzero(~predicted & actual & valid_mask))
            frames = int(torch.count_nonzero(valid_mask))
            total_frames += frames
            total_loss += float(loss) * frames
            for name in component_totals:
                component_totals[name] += float(components[name]) * frames
            total_gradient_norm += float(gradient_norm)
            optimizer_steps += 1

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    metrics = {
        "loss": total_loss / max(1, total_frames),
        "precision_at_0_5": precision,
        "recall_at_0_5": recall,
        "f1_at_0_5": 2.0 * precision * recall / max(1e-12, precision + recall),
        "gradient_norm": total_gradient_norm / max(1, optimizer_steps),
    }
    metrics.update(
        {
            f"loss_{name}": value / max(1, total_frames)
            for name, value in component_totals.items()
        }
    )
    return metrics


@torch.inference_mode()
def evaluate(
    model: RenderAwareDtd,
    loader: DataLoader,
    device: torch.device,
    event_threshold: float = 0.55,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    if not 0.0 < event_threshold < 1.0:
        raise ValueError("event_threshold must be in (0, 1)")
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    event_probabilities: list[np.ndarray] = []
    event_targets: list[np.ndarray] = []
    event_valid_masks: list[np.ndarray] = []
    event_reset_masks: list[np.ndarray] = []
    event_modes: list[str] = []
    event_cluster_ids: list[str] = []
    strata: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    total_loss = 0.0
    total_frames = 0
    for batch in loader:
        microphone = batch["microphone"].to(device, non_blocking=True)
        render = batch["render"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        valid_mask = batch.get("valid_mask")
        valid_mask = (
            torch.ones_like(target, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.to(device, non_blocking=True, dtype=torch.bool)
        )
        reset_before = batch.get("reset_before")
        reset_before = (
            torch.zeros_like(target, dtype=torch.bool)
            if reset_before is None
            else reset_before.to(device, non_blocking=True, dtype=torch.bool)
        )
        logits, _ = model(microphone, render)
        logits = apply_stream_resets(
            model,
            logits,
            microphone,
            render,
            reset_before,
            model.config.hop_length,
        )
        frame_loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        loss = (frame_loss * valid_mask).sum()
        total_loss += float(loss)
        total_frames += int(torch.count_nonzero(valid_mask))
        batch_probability = torch.sigmoid(logits).cpu().numpy()
        batch_target = target.cpu().numpy()
        batch_valid = valid_mask.cpu().numpy().astype(bool, copy=False)
        batch_reset = reset_before.cpu().numpy().astype(bool, copy=False)
        probabilities.append(batch_probability[batch_valid])
        targets.append(batch_target[batch_valid])
        sources = batch.get("source", ["aec_legacy"] * batch_probability.shape[0])
        modes = batch.get("mode", ["mixture"] * batch_probability.shape[0])
        clusters = batch.get("cluster_id", batch.get("file_id"))
        if clusters is None:
            clusters = [f"stream:{index}" for index in range(len(modes))]
        elif isinstance(clusters, torch.Tensor):
            clusters = clusters.cpu().tolist()
        for index, (source, mode, cluster_id) in enumerate(
            zip(sources, modes, clusters)
        ):
            valid = batch_valid[index]
            pair = (
                batch_probability[index][valid],
                batch_target[index][valid],
            )
            strata[f"source:{source}"].append(pair)
            strata[f"mode:{mode}"].append(pair)
            event_probabilities.append(batch_probability[index])
            event_targets.append(batch_target[index])
            event_valid_masks.append(valid)
            event_reset_masks.append(batch_reset[index])
            event_modes.append(str(mode))
            event_cluster_ids.append(str(cluster_id))

    probability = np.concatenate(probabilities)
    target = np.concatenate(targets)
    threshold, metrics = select_threshold(probability, target)
    metrics["loss"] = total_loss / max(1, total_frames)
    metrics["selected_threshold"] = threshold
    metrics["low_fpr_operating_points"] = low_fpr_operating_points(probability, target)
    metrics["event"] = event_metrics(
        event_probabilities,
        event_targets,
        event_threshold,
        modes=event_modes,
        cluster_ids=event_cluster_ids,
        valid_masks=event_valid_masks,
        reset_masks=event_reset_masks,
    )
    metrics["strata"] = {}
    for key, values in sorted(strata.items()):
        stratum_probability = np.concatenate([value[0] for value in values])
        stratum_target = np.concatenate([value[1] for value in values])
        metrics["strata"][key] = binary_metrics(
            stratum_probability,
            stratum_target,
            threshold,
        )
    return metrics, probability, target


def atomic_torch_save(
    payload: dict[str, object],
    path: Path,
    *,
    replace: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite checkpoint: {path}"
                ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                json_safe(value),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    model: RenderAwareDtd,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    threshold: float,
    validation_metrics: dict[str, object],
    args: argparse.Namespace,
    selection_score: float,
    history: list[dict[str, object]],
    *,
    replace: bool = True,
) -> None:
    atomic_torch_save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "model_state": model.state_dict(),
            "model_config": model.config.to_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "epoch": epoch,
            "decision_threshold": threshold,
            "validation_metrics": validation_metrics,
            "selection_score": selection_score,
            "training_args": dict(vars(args)),
            "history": json_safe(history),
        },
        path,
        replace=replace,
    )


def validate_resume_history(
    raw: object,
    start_epoch: int,
    source: str,
) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        raise ValueError(f"training history must contain a JSON array: {source}")
    if not all(isinstance(record, dict) and "epoch" in record for record in raw):
        raise ValueError(f"training history contains an invalid record: {source}")
    history = list(raw)
    epochs = [int(record["epoch"]) for record in history]
    expected = list(range(start_epoch))
    if epochs != expected:
        raise ValueError(
            "training history epochs do not match the resume checkpoint: "
            f"source={source}, expected={expected}, actual={epochs}"
        )
    return history


def load_resume_history(
    path: Path,
    start_epoch: int,
    checkpoint: dict[str, object],
) -> list[dict[str, object]]:
    embedded = checkpoint.get("history")
    if embedded is not None:
        return validate_resume_history(embedded, start_epoch, "checkpoint")
    if not path.is_file():
        raise FileNotFoundError(
            "legacy resume checkpoint has no embedded history and the "
            f"external history is missing: {path}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return validate_resume_history(raw, start_epoch, str(path))


RESUME_ARGUMENT_ALLOWLIST = frozenset({"resume", "output_dir", "num_workers"})
LEGACY_TRAINING_ARGUMENT_DEFAULTS: dict[str, object] = {
    "lr_schedule": "constant",
    "warmup_epochs": 0,
    "min_learning_rate": 1e-5,
    "model_size": 192,
    "gru_layers": 2,
    "frequency_bands": 48,
    "dropout": 0.1,
    "wenet_far_only_probability": 0.40,
    "hard_negative_loss_weight": 0.0,
    "onset_loss_weight": 0.0,
    "event_loss_weight": 0.0,
    "reset_loss_weight": 0.0,
    "hard_negative_top_fraction": 0.10,
    "event_decision_threshold": 0.55,
    "event_temperature": 0.08,
    "v5_reset_probability": 0.0,
    "v5_hard_far_only_probability": 0.0,
    "v5_maximum_gap_hops": 8,
    "v5_validation_reset_probability": 0.0,
    "v5_validation_hard_far_only_probability": 0.0,
    "v5_validation_maximum_gap_hops": 8,
}
TRAINING_ARTIFACT_NAMES = (
    "best.pt",
    "last.pt",
    "history.json",
    "training_summary.json",
)


def normalized_training_args(values: dict[str, object]) -> dict[str, object]:
    normalized = dict(values)
    for name, default in LEGACY_TRAINING_ARGUMENT_DEFAULTS.items():
        normalized.setdefault(name, default)
    return normalized


def validate_resume_recipe(
    checkpoint: dict[str, object],
    args: argparse.Namespace,
) -> None:
    stored_raw = checkpoint.get("training_args")
    if not isinstance(stored_raw, dict):
        raise ValueError("resume checkpoint has no valid training_args")
    stored = normalized_training_args(stored_raw)
    current = normalized_training_args(dict(vars(args)))
    differences: list[str] = []
    for name in sorted(set(stored) | set(current)):
        if name in RESUME_ARGUMENT_ALLOWLIST:
            continue
        old = stored.get(name, "<missing>")
        new = current.get(name, "<missing>")
        if old != new:
            differences.append(f"{name}: checkpoint={old!r}, current={new!r}")
    if differences:
        raise ValueError(
            "resume training recipe mismatch; only --num-workers, "
            "--output-dir, and --resume may change:\n" + "\n".join(differences)
        )


def assert_fresh_output_directory(output_dir: Path) -> None:
    conflicts = [
        output_dir / name
        for name in TRAINING_ARTIFACT_NAMES
        if (output_dir / name).exists()
    ]
    checkpoint_directory = output_dir / "checkpoints"
    if checkpoint_directory.is_dir():
        conflicts.extend(sorted(checkpoint_directory.glob("epoch-*.pt")))
    if conflicts:
        formatted = "\n".join(str(path) for path in conflicts)
        raise FileExistsError(
            "fresh training refuses to overwrite existing artifacts:\n" + formatted
        )


def load_training_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "epoch" not in checkpoint:
        raise ValueError(f"invalid training checkpoint: {path}")
    return checkpoint


MUTABLE_CHECKPOINT_LOAD_ERRORS = (
    EOFError,
    OSError,
    RuntimeError,
    ValueError,
    pickle.UnpicklingError,
)


def try_load_mutable_checkpoint(
    path: Path,
) -> tuple[dict[str, object] | None, Exception | None]:
    if not path.is_file():
        return None, None
    try:
        return load_training_checkpoint(path), None
    except MUTABLE_CHECKPOINT_LOAD_ERRORS as error:
        return None, error


def archive_unreadable_checkpoint(path: Path) -> Path:
    for index in range(1, 10_000):
        archive = path.with_name(f"{path.name}.corrupt-{index:03d}")
        try:
            os.link(path, archive)
        except FileExistsError:
            continue
        path.unlink()
        return archive
    raise RuntimeError(f"cannot allocate a corrupt-checkpoint archive name: {path}")


def validate_resume_checkpoint_format(
    checkpoint: dict[str, object],
    source: Path | str,
) -> None:
    version = checkpoint.get("format_version")
    if not isinstance(version, int):
        raise ValueError(f"resume checkpoint has no valid format version: {source}")
    if version < 3:
        raise ValueError(
            "checkpoint format cannot be resumed safely because its selection "
            "score policy is incompatible with the current trainer: "
            f"source={source}, format_version={version}"
        )
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "checkpoint format is newer than this trainer: "
            f"source={source}, format_version={version}, "
            f"maximum={CHECKPOINT_FORMAT_VERSION}"
        )
    policy = checkpoint.get("selection_policy_version")
    if version == 3 and policy is None:
        return
    if policy != SELECTION_POLICY_VERSION:
        raise ValueError(
            "checkpoint selection policy is incompatible with this trainer: "
            f"source={source}, policy={policy!r}, "
            f"expected={SELECTION_POLICY_VERSION!r}"
        )


def immutable_checkpoint_epoch(path: Path) -> int:
    value = path.stem.removeprefix("epoch-")
    if not value.isdigit():
        raise ValueError(f"invalid immutable checkpoint name: {path}")
    return int(value)


def latest_immutable_checkpoint(output_dir: Path) -> Path | None:
    paths = list((output_dir / "checkpoints").glob("epoch-*.pt"))
    if not paths:
        return None
    return max(paths, key=immutable_checkpoint_epoch)


def resolve_resume_checkpoint(
    output_dir: Path,
    requested_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, object], list[str]]:
    actions: list[str] = []
    requested_error: Exception | None = None
    if requested_path.name.lower() == "last.pt":
        requested_checkpoint, requested_error = try_load_mutable_checkpoint(
            requested_path
        )
    else:
        requested_checkpoint = (
            load_training_checkpoint(requested_path)
            if requested_path.is_file()
            else None
        )
    if requested_checkpoint is not None:
        validate_resume_checkpoint_format(
            requested_checkpoint,
            requested_path,
        )
        validate_resume_recipe(requested_checkpoint, args)
        requested_epoch = int(requested_checkpoint["epoch"])
    elif requested_path.name.lower() == "last.pt":
        requested_epoch = -1
    else:
        raise FileNotFoundError(f"resume checkpoint does not exist: {requested_path}")

    latest_path = latest_immutable_checkpoint(output_dir)
    if latest_path is None:
        if requested_error is not None:
            raise ValueError(
                "last.pt is unreadable and no immutable epoch checkpoint "
                f"exists: {requested_path}"
            ) from requested_error
        if requested_checkpoint is None:
            raise FileNotFoundError(
                "last.pt is missing and no immutable epoch checkpoint exists: "
                f"{output_dir}"
            )
        return requested_path, requested_checkpoint, actions

    latest_checkpoint = load_training_checkpoint(latest_path)
    validate_resume_checkpoint_format(latest_checkpoint, latest_path)
    latest_epoch = int(latest_checkpoint["epoch"])
    filename_epoch = immutable_checkpoint_epoch(latest_path)
    if latest_epoch != filename_epoch:
        raise ValueError(
            "immutable checkpoint epoch does not match its filename: "
            f"path={latest_path}, payload_epoch={latest_epoch}"
        )
    validate_resume_recipe(latest_checkpoint, args)
    if requested_error is not None:
        archive = archive_unreadable_checkpoint(requested_path)
        actions.append(
            "archived unreadable last.pt as "
            f"{archive.name} ({type(requested_error).__name__})"
        )
    if latest_epoch <= requested_epoch:
        if requested_checkpoint is None:
            raise AssertionError("missing requested checkpoint has no fallback")
        return requested_path, requested_checkpoint, actions
    if requested_path.name.lower() != "last.pt":
        raise ValueError(
            "newer immutable checkpoints would collide with an explicit "
            "rollback; resume from last.pt or the newest epoch checkpoint: "
            f"requested_epoch={requested_epoch}, latest_epoch={latest_epoch}"
        )
    return latest_path, latest_checkpoint, actions


def history_record_selection_score(record: dict[str, object]) -> float:
    raw_score = record.get("selection_score")
    if raw_score is None:
        validation = record.get("validation")
        if not isinstance(validation, dict):
            raise ValueError("history validation entry must be an object")
        raw_score = validation_selection_score(validation)
    score = float(raw_score)
    if not math.isfinite(score):
        raise ValueError("history selection score must be finite")
    return score


def checkpoint_selection_score(checkpoint: dict[str, object]) -> float:
    if "selection_score" in checkpoint:
        score = float(checkpoint["selection_score"])
    else:
        validation = checkpoint.get("validation_metrics")
        if not isinstance(validation, dict):
            raise ValueError(
                "checkpoint has neither selection_score nor validation_metrics"
            )
        score = validation_selection_score(validation)
    if not math.isfinite(score):
        raise ValueError("checkpoint selection score must be finite")
    return score


def reconcile_resume_artifacts(
    output_dir: Path,
    resume_checkpoint: dict[str, object],
    history: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[str]]:
    resume_epoch = int(resume_checkpoint["epoch"])
    if resume_epoch < 0 or len(history) != resume_epoch + 1:
        raise ValueError(
            "resume checkpoint and embedded history length are inconsistent: "
            f"epoch={resume_epoch}, history={len(history)}"
        )
    validate_resume_recipe(resume_checkpoint, args)
    validate_resume_checkpoint_format(resume_checkpoint, "resolved resume")
    actions: list[str] = []
    checkpoint_directory = output_dir / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    canonical_path = checkpoint_directory / f"epoch-{resume_epoch:03d}.pt"
    if canonical_path.is_file():
        canonical = load_training_checkpoint(canonical_path)
        validate_resume_checkpoint_format(canonical, canonical_path)
        validate_resume_recipe(canonical, args)
        if int(canonical["epoch"]) != resume_epoch:
            raise ValueError(
                "canonical immutable checkpoint has the wrong epoch: "
                f"{canonical_path}"
            )
    else:
        atomic_torch_save(resume_checkpoint, canonical_path, replace=False)
        actions.append(f"restored immutable checkpoint {canonical_path.name}")

    last_path = output_dir / "last.pt"
    last_checkpoint, last_error = try_load_mutable_checkpoint(last_path)
    if last_error is not None:
        archive = archive_unreadable_checkpoint(last_path)
        actions.append(
            "archived unreadable last.pt as "
            f"{archive.name} ({type(last_error).__name__})"
        )
    if last_checkpoint is not None:
        validate_resume_checkpoint_format(last_checkpoint, last_path)
        validate_resume_recipe(last_checkpoint, args)
        last_epoch = int(last_checkpoint["epoch"])
        if last_epoch > resume_epoch:
            raise ValueError(
                "last.pt is newer than the resolved resume checkpoint: "
                f"last_epoch={last_epoch}, resume_epoch={resume_epoch}"
            )
    else:
        last_epoch = -1
    if last_epoch < resume_epoch:
        atomic_torch_save(resume_checkpoint, last_path)
        actions.append(f"advanced last.pt to epoch {resume_epoch}")

    expected_best_record = max(history, key=history_record_selection_score)
    expected_best_epoch = int(expected_best_record["epoch"])
    expected_best_score = history_record_selection_score(expected_best_record)
    best_path = output_dir / "best.pt"
    best_checkpoint, best_error = try_load_mutable_checkpoint(best_path)
    if best_error is not None:
        archive = archive_unreadable_checkpoint(best_path)
        actions.append(
            "archived unreadable best.pt as "
            f"{archive.name} ({type(best_error).__name__})"
        )
    repair_best = best_checkpoint is None
    if best_checkpoint is not None:
        validate_resume_checkpoint_format(best_checkpoint, best_path)
        validate_resume_recipe(best_checkpoint, args)
        best_epoch = int(best_checkpoint["epoch"])
        best_score = checkpoint_selection_score(best_checkpoint)
        if best_epoch > resume_epoch or best_score > expected_best_score + 1e-12:
            raise ValueError(
                "best.pt is inconsistent with the resolved resume history: "
                f"best_epoch={best_epoch}, best_score={best_score}, "
                f"expected_epoch={expected_best_epoch}, "
                f"expected_score={expected_best_score}"
            )
        repair_best = best_epoch != expected_best_epoch or not math.isclose(
            best_score,
            expected_best_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    if repair_best:
        source_path = (
            canonical_path
            if expected_best_epoch == resume_epoch
            else checkpoint_directory / f"epoch-{expected_best_epoch:03d}.pt"
        )
        source_checkpoint = (
            resume_checkpoint
            if expected_best_epoch == resume_epoch
            else load_training_checkpoint(source_path)
        )
        validate_resume_checkpoint_format(source_checkpoint, source_path)
        validate_resume_recipe(source_checkpoint, args)
        if int(source_checkpoint["epoch"]) != expected_best_epoch:
            raise ValueError(
                "best-checkpoint recovery source has the wrong epoch: " f"{source_path}"
            )
        source_score = checkpoint_selection_score(source_checkpoint)
        if not math.isclose(
            source_score,
            expected_best_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "best-checkpoint recovery source score does not match history: "
                f"source={source_score}, expected={expected_best_score}"
            )
        atomic_torch_save(source_checkpoint, best_path)
        best_checkpoint = source_checkpoint
        actions.append(f"restored best.pt from epoch {expected_best_epoch}")
    if best_checkpoint is None:
        raise AssertionError("best checkpoint reconciliation did not produce best.pt")
    return best_checkpoint, actions


def reconstruct_early_stopping(
    history: list[dict[str, object]],
    min_delta: float,
) -> tuple[float, int]:
    best_score = -1.0
    epochs_without_improvement = 0
    for record in history:
        validation = record["validation"]
        if not isinstance(validation, dict):
            raise ValueError("history validation entry must be an object")
        score = validation_selection_score(validation)
        if score > best_score + min_delta:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return best_score, epochs_without_improvement


def early_stopping_exhausted(
    patience: int,
    epochs_without_improvement: int,
) -> bool:
    return patience > 0 and epochs_without_improvement >= patience


def remaining_training_epochs(
    start_epoch: int,
    total_epochs: int,
    patience: int,
    epochs_without_improvement: int,
) -> range:
    stop = (
        start_epoch
        if early_stopping_exhausted(
            patience,
            epochs_without_improvement,
        )
        else total_epochs
    )
    return range(start_epoch, stop)


def validation_selection_score(metrics: dict[str, object]) -> float:
    low_fpr = metrics["low_fpr_operating_points"]
    if not isinstance(low_fpr, dict):
        raise ValueError("validation low-FPR metrics must be an object")
    point_one_percent = low_fpr["0.100%"]
    half_percent = low_fpr["0.500%"]
    one_percent = low_fpr["1.000%"]
    if not all(
        isinstance(point, dict)
        for point in (point_one_percent, half_percent, one_percent)
    ):
        raise ValueError("validation low-FPR metric points must be objects")
    return (
        float(metrics["average_precision"])
        + 0.10 * float(metrics["f1"])
        + 0.10 * float(point_one_percent["recall"])
        + 0.15 * float(half_percent["recall"])
        + 0.25 * float(one_percent["recall"])
    )


def json_safe(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("--warmup-epochs must be in [0, epochs)")
    if not 0.0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("--min-learning-rate must be in [0, learning-rate]")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta cannot be negative")
    if args.examples_per_epoch <= 0:
        raise ValueError("--examples-per-epoch must be positive")
    if args.validation_examples <= 0:
        raise ValueError("--validation-examples must be positive")
    if args.positive_weight <= 0.0:
        raise ValueError("--positive-weight must be positive")
    if not 0.0 <= args.wenet_far_only_probability <= 0.90:
        raise ValueError("--wenet-far-only-probability must be in [0, 0.90]")
    objective_config = EventObjectiveConfig(
        hard_negative_weight=args.hard_negative_loss_weight,
        onset_weight=args.onset_loss_weight,
        event_weight=args.event_loss_weight,
        reset_weight=args.reset_loss_weight,
        hard_negative_top_fraction=args.hard_negative_top_fraction,
        decision_threshold=args.event_decision_threshold,
        event_temperature=args.event_temperature,
    )
    objective_config.validate()
    if not 0.0 <= args.v5_reset_probability <= 1.0:
        raise ValueError("--v5-reset-probability must be in [0, 1]")
    if not 0.0 <= args.v5_hard_far_only_probability <= 1.0:
        raise ValueError("--v5-hard-far-only-probability must be in [0, 1]")
    if args.v5_maximum_gap_hops <= 0:
        raise ValueError("--v5-maximum-gap-hops must be positive")
    if not 0.0 <= args.v5_validation_reset_probability <= 1.0:
        raise ValueError("--v5-validation-reset-probability must be in [0, 1]")
    if not 0.0 <= args.v5_validation_hard_far_only_probability <= 1.0:
        raise ValueError("--v5-validation-hard-far-only-probability must be in [0, 1]")
    if args.v5_validation_maximum_gap_hops <= 0:
        raise ValueError("--v5-validation-maximum-gap-hops must be positive")
    dataset_weights = (
        args.aec_narrow_weight,
        args.aec_fullband_weight,
        args.wenet_weight,
        args.aec_real_weight,
    )
    if not all(math.isfinite(weight) for weight in dataset_weights):
        raise ValueError("dataset weights must be finite")
    if any(weight < 0.0 for weight in dataset_weights):
        raise ValueError("dataset weights must be non-negative")
    if sum(dataset_weights) <= 0.0:
        raise ValueError("dataset weights must sum to a positive value")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint: dict[str, object] | None = None
    resolved_resume_path: Path | None = None
    resume_resolution_actions: list[str] = []
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if output_dir != resume_path.parent and output_dir not in resume_path.parents:
            raise ValueError(
                "--resume must reference a checkpoint inside --output-dir: "
                f"resume={resume_path}, output={output_dir}"
            )
        (
            resolved_resume_path,
            resume_checkpoint,
            resume_resolution_actions,
        ) = resolve_resume_checkpoint(output_dir, resume_path, args)
        requested_model_config = DtdModelConfig(
            projection_size=args.model_size,
            hidden_size=args.model_size,
            gru_layers=args.gru_layers,
            frequency_bands=args.frequency_bands,
            dropout=args.dropout,
        ).to_dict()
        if resume_checkpoint.get("model_config") != requested_model_config:
            raise ValueError(
                "resume model_config does not match the current CLI recipe: "
                f"checkpoint={resume_checkpoint.get('model_config')}, "
                f"current={requested_model_config}"
            )
    else:
        assert_fresh_output_directory(output_dir)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    configure_cuda_backend(device)

    dataset_description: dict[str, object]
    if args.enriched_root:
        train_dataset, validation_dataset, dataset_description = (
            create_enriched_datasets(
                args.enriched_root,
                sequence_seconds=args.sequence_seconds,
                examples_per_epoch=args.examples_per_epoch,
                validation_examples=args.validation_examples,
                seed=args.seed,
                dataset_weights=dataset_weights,
                wenet_far_only_probability=(args.wenet_far_only_probability),
                v5_reset_probability=args.v5_reset_probability,
                v5_hard_far_only_probability=(args.v5_hard_far_only_probability),
                v5_maximum_gap_hops=args.v5_maximum_gap_hops,
                v5_validation_reset_probability=(args.v5_validation_reset_probability),
                v5_validation_hard_far_only_probability=(
                    args.v5_validation_hard_far_only_probability
                ),
                v5_validation_maximum_gap_hops=(args.v5_validation_maximum_gap_hops),
                train_limit=args.train_limit,
                validation_limit=args.validation_limit,
            )
        )
    else:
        train_dataset = AecChallengeDtdDataset(
            args.dataset_root,
            "train",
            sequence_seconds=args.sequence_seconds,
            augment=True,
            seed=args.seed,
            validation_modulo=args.validation_modulo,
            limit=args.train_limit,
        )
        validation_dataset = AecChallengeDtdDataset(
            args.dataset_root,
            "validation",
            sequence_seconds=None,
            augment=False,
            seed=args.seed,
            validation_modulo=args.validation_modulo,
            limit=args.validation_limit,
        )
        dataset_description = {
            "type": "aec_challenge_synthetic",
            "root": str(Path(args.dataset_root).resolve()),
        }
    pin_memory = device.type == "cuda"
    validation_loader = create_loader(
        validation_dataset,
        batch_size=args.batch_size if args.enriched_root else 1,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=pin_memory,
    )

    model_config = DtdModelConfig(
        projection_size=args.model_size,
        hidden_size=args.model_size,
        gru_layers=args.gru_layers,
        frequency_bands=args.frequency_bands,
        dropout=args.dropout,
    )
    if resume_checkpoint is not None:
        model_config = DtdModelConfig(**resume_checkpoint["model_config"])
    model = RenderAwareDtd(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if resume_checkpoint is not None:
        if "scaler_state" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        if "random_state" in resume_checkpoint:
            random.setstate(resume_checkpoint["random_state"])
        if "numpy_random_state" in resume_checkpoint:
            np.random.set_state(resume_checkpoint["numpy_random_state"])
        if "torch_random_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["torch_random_state"])
        cuda_random_state = resume_checkpoint.get("cuda_random_state")
        if cuda_random_state is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_random_state)
    resource_monitor = ResourceMonitor.create()
    training_started = time.perf_counter()
    print(
        json.dumps(
            {
                "device": str(device),
                "train_examples": len(train_dataset),
                "validation_examples": len(validation_dataset),
                "dataset": dataset_description,
                "complexity": model.complexity().to_dict(),
                "model_config": model.config.to_dict(),
                "event_objective": objective_config.__dict__,
                "resolved_resume": (
                    str(resolved_resume_path)
                    if resolved_resume_path is not None
                    else None
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    history_path = output_dir / "history.json"
    history = (
        load_resume_history(history_path, start_epoch, resume_checkpoint)
        if resume_checkpoint is not None
        else []
    )
    if resume_checkpoint is not None:
        atomic_write_json(history_path, history)
    best_path = output_dir / "best.pt"
    checkpoint_directory = output_dir / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is not None:
        best_checkpoint, reconciliation_actions = reconcile_resume_artifacts(
            output_dir,
            resume_checkpoint,
            history,
            args,
        )
        for recovery_action in resume_resolution_actions + reconciliation_actions:
            print(
                json.dumps(
                    {"resume_recovery": recovery_action},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        best_selection_score = checkpoint_selection_score(best_checkpoint)
    else:
        best_selection_score = -1.0
    early_stopping_best, epochs_without_improvement = reconstruct_early_stopping(
        history,
        args.early_stopping_min_delta,
    )
    stopped_early = early_stopping_exhausted(
        args.early_stopping_patience,
        epochs_without_improvement,
    )
    if stopped_early:
        print(
            json.dumps(
                {
                    "early_stopping": True,
                    "resumed_after_stop": True,
                    "epoch": start_epoch - 1,
                    "best_material_selection_score": early_stopping_best,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for epoch in remaining_training_epochs(
        start_epoch,
        args.epochs,
        args.early_stopping_patience,
        epochs_without_improvement,
    ):
        started = time.perf_counter()
        train_dataset.set_epoch(epoch)
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(args.seed + epoch * 10_000_019)
        train_loader = create_loader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            generator=shuffle_generator,
        )
        learning_rate = learning_rate_for_epoch(args, epoch)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        train_metrics = run_training_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.positive_weight,
            objective_config,
            resource_monitor,
        )
        validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            device,
            objective_config.decision_threshold,
        )
        elapsed = time.perf_counter() - started
        selection_score = validation_selection_score(validation_metrics)
        if selection_score > early_stopping_best + args.early_stopping_min_delta:
            early_stopping_best = selection_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record: dict[str, object] = {
            "epoch": epoch,
            "seconds": elapsed,
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_score": selection_score,
            "early_stopping": {
                "monitor": (
                    "validation AP + 0.10*F1 + 0.10*recall@0.1%FPR "
                    "+ 0.15*recall@0.5%FPR + 0.25*recall@1%FPR"
                ),
                "best_material_score": early_stopping_best,
                "epochs_without_improvement": epochs_without_improvement,
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
            },
        }
        history.append(record)
        # Checkpoints carry the frozen production contract, not a per-candidate
        # frame-F1 threshold that would make event comparisons non-transferable.
        threshold = objective_config.decision_threshold
        save_checkpoint(
            checkpoint_directory / f"epoch-{epoch:03d}.pt",
            model,
            optimizer,
            scaler,
            epoch,
            threshold,
            validation_metrics,
            args,
            selection_score,
            history,
            replace=False,
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            threshold,
            validation_metrics,
            args,
            selection_score,
            history,
        )
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scaler,
                epoch,
                threshold,
                validation_metrics,
                args,
                selection_score,
                history,
            )
        atomic_write_json(history_path, history)
        print(
            json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False),
            flush=True,
        )
        if early_stopping_exhausted(
            args.early_stopping_patience,
            epochs_without_improvement,
        ):
            stopped_early = True
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "best_material_selection_score": early_stopping_best,
                        "epochs_without_improvement": (epochs_without_improvement),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    final_metrics, probability, target = evaluate(
        model,
        validation_loader,
        device,
        float(best["decision_threshold"]),
    )
    fixed_metrics = binary_metrics(
        probability,
        target,
        float(best["decision_threshold"]),
    )
    summary = {
        "best_epoch": int(best["epoch"]),
        "decision_threshold": float(best["decision_threshold"]),
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "early_stopping": {
            "monitor": (
                "validation AP + 0.10*F1 + 0.10*recall@0.1%FPR "
                "+ 0.15*recall@0.5%FPR + 0.25*recall@1%FPR"
            ),
            "patience": args.early_stopping_patience,
            "min_delta": args.early_stopping_min_delta,
            "best_material_score": early_stopping_best,
            "epochs_without_improvement": epochs_without_improvement,
        },
        "validation": final_metrics,
        "validation_at_checkpoint_threshold": fixed_metrics,
        "selection_score": float(best["selection_score"]),
        "training_args": vars(args),
        "model_config": model.config.to_dict(),
        "complexity": model.complexity().to_dict(),
        "dataset": dataset_description,
        "runtime": {
            "training_seconds_this_invocation": (
                time.perf_counter() - training_started
            ),
            "epoch_seconds_total": sum(float(record["seconds"]) for record in history),
            "resources": resource_monitor.to_dict(device),
        },
    }
    atomic_write_json(output_dir / "training_summary.json", summary)
    print(
        json.dumps(json_safe(summary), ensure_ascii=False, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
