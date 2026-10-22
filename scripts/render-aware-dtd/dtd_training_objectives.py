from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class StreamingDtdModel(Protocol):
    def __call__(
        self,
        microphone: torch.Tensor,
        render: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class EventObjectiveConfig:
    """Device-independent, production-timing-aware training parameters."""

    hard_negative_weight: float = 0.0
    onset_weight: float = 0.0
    event_weight: float = 0.0
    reset_weight: float = 0.0
    hard_negative_top_fraction: float = 0.10
    onset_hops: int = 18
    smoothing_hops: int = 3
    evidence_window_hops: int = 18
    required_evidence_hops: int = 9
    tail_hops: int = 4
    decision_threshold: float = 0.55
    event_temperature: float = 0.08
    reset_warmup_hops: int = 18

    def validate(self) -> None:
        for name in (
            "hard_negative_weight",
            "onset_weight",
            "event_weight",
            "reset_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < self.hard_negative_top_fraction <= 1.0:
            raise ValueError("hard_negative_top_fraction must be in (0, 1]")
        for name in (
            "onset_hops",
            "smoothing_hops",
            "evidence_window_hops",
            "required_evidence_hops",
            "tail_hops",
            "reset_warmup_hops",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.required_evidence_hops > self.evidence_window_hops:
            raise ValueError(
                "required_evidence_hops cannot exceed evidence_window_hops"
            )
        if self.tail_hops > self.evidence_window_hops:
            raise ValueError("tail_hops cannot exceed evidence_window_hops")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be in (0, 1)")
        if not math.isfinite(self.event_temperature) or self.event_temperature <= 0.0:
            raise ValueError("event_temperature must be finite and positive")


def weighted_frame_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    frame_loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    weights = 1.0 + target * (positive_weight - 1.0)
    valid = valid_mask.to(dtype=frame_loss.dtype)
    return (frame_loss * weights * valid).sum() / valid.sum().clamp_min(1.0)


def apply_stream_resets(
    model: StreamingDtdModel,
    logits: torch.Tensor,
    microphone: torch.Tensor,
    render: torch.Tensor,
    reset_before: torch.Tensor,
    hop_length: int,
) -> torch.Tensor:
    """Replace post-boundary logits with inference from a cleared state."""

    reset_locations = torch.nonzero(
        reset_before[:, 1:],
        as_tuple=False,
    )
    if reset_locations.numel() == 0:
        return logits
    suffixes: list[tuple[int, int, int]] = []
    maximum_samples = 0
    for batch_index, reset_offset in reset_locations.tolist():
        reset_frame = reset_offset + 1
        start_sample = reset_frame * hop_length
        sample_count = microphone.shape[1] - start_sample
        if sample_count < 2 * hop_length:
            continue
        suffixes.append((batch_index, reset_frame, sample_count))
        maximum_samples = max(maximum_samples, sample_count)
    if not suffixes:
        return logits

    reset_microphone = microphone.new_zeros((len(suffixes), maximum_samples))
    reset_render = render.new_zeros((len(suffixes), maximum_samples))
    for suffix_index, (batch_index, reset_frame, sample_count) in enumerate(suffixes):
        start_sample = reset_frame * hop_length
        reset_microphone[suffix_index, :sample_count] = microphone[
            batch_index,
            start_sample : start_sample + sample_count,
        ]
        reset_render[suffix_index, :sample_count] = render[
            batch_index,
            start_sample : start_sample + sample_count,
        ]

    suffix_logits, _ = model(reset_microphone, reset_render)
    reset_logits = logits.clone()
    for suffix_index, (batch_index, reset_frame, sample_count) in enumerate(suffixes):
        actual_frames = 1 + (sample_count - 2 * hop_length) // hop_length
        available = min(
            actual_frames,
            logits.shape[1] - reset_frame,
        )
        reset_logits[
            batch_index,
            reset_frame : reset_frame + available,
        ] = suffix_logits[suffix_index, :available]
    return reset_logits


def _causal_mean(values: torch.Tensor, hops: int) -> torch.Tensor:
    if hops == 1:
        return values
    prefix = F.pad(values.unsqueeze(1), (hops - 1, 0))
    summed = F.avg_pool1d(prefix, kernel_size=hops, stride=1) * hops
    counts = torch.arange(
        1,
        values.shape[1] + 1,
        device=values.device,
        dtype=values.dtype,
    ).clamp_max(hops)
    return summed.squeeze(1) / counts


def _rolling_mean(values: torch.Tensor, hops: int) -> torch.Tensor:
    prefix = F.pad(values.unsqueeze(1), (hops - 1, 0))
    return F.avg_pool1d(prefix, kernel_size=hops, stride=1).squeeze(1)


def _soft_max(values: torch.Tensor, temperature: float) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError("soft maximum requires at least one value")
    weights = torch.softmax(values / temperature, dim=-1)
    return torch.sum(weights * values, dim=-1)


def _valid_segments_from_values(
    valid_mask: list[bool],
    reset_before: list[bool],
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, (valid, reset) in enumerate(zip(valid_mask, reset_before)):
        if (reset or not valid) and start is not None:
            segments.append((start, index))
            start = None
        if valid and start is None:
            start = index
    if start is not None:
        segments.append((start, len(valid_mask)))
    return segments


def hard_negative_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    reset_before: torch.Tensor,
    far_only_mask: torch.Tensor,
    config: EventObjectiveConfig,
) -> torch.Tensor:
    """Penalize the high-score tail and persistent evidence on far-only streams."""

    losses: list[torch.Tensor] = []
    probabilities = torch.sigmoid(logits)
    valid_values = valid_mask.detach().cpu().tolist()
    reset_values = reset_before.detach().cpu().tolist()
    for index in torch.nonzero(far_only_mask, as_tuple=False).flatten().tolist():
        for start, stop in _valid_segments_from_values(
            valid_values[index],
            reset_values[index],
        ):
            negative = target[index, start:stop] < 0.5
            selected_logits = logits[index, start:stop][negative]
            if selected_logits.numel() == 0:
                continue
            count = max(
                1,
                math.ceil(selected_logits.numel() * config.hard_negative_top_fraction),
            )
            top_logits = torch.topk(selected_logits, count).values
            tail_loss = F.softplus(top_logits).mean()

            selected_probability = probabilities[index, start:stop][negative].unsqueeze(
                0
            )
            smoothed = _causal_mean(
                selected_probability,
                config.smoothing_hops,
            )
            persistent = _rolling_mean(
                smoothed,
                config.required_evidence_hops,
            ).squeeze(0)
            persistent_peak = _soft_max(
                persistent,
                config.event_temperature,
            )
            persistence_loss = -torch.log1p(-persistent_peak.clamp(max=1.0 - 1e-6))
            losses.append(0.5 * (tail_loss + persistence_loss))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def onset_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    reset_before: torch.Tensor,
    config: EventObjectiveConfig,
) -> torch.Tensor:
    """Reward fast post-onset confidence without rewarding pre-onset triggers."""

    losses: list[torch.Tensor] = []
    active = (target >= 0.5) & valid_mask
    previous = torch.zeros_like(active)
    previous[:, 1:] = active[:, :-1]
    previous[reset_before] = False
    onsets = torch.nonzero(active & ~previous, as_tuple=False)
    active_values = active.detach().cpu().tolist()
    reset_values = reset_before.detach().cpu().tolist()
    onset_weights = torch.linspace(
        1.0,
        0.25,
        steps=config.onset_hops,
        device=logits.device,
        dtype=logits.dtype,
    )
    for batch_index, onset in onsets.tolist():
        limit = min(target.shape[1], onset + config.onset_hops)
        stop = onset
        while (
            stop < limit
            and active_values[batch_index][stop]
            and (stop == onset or not reset_values[batch_index][stop])
        ):
            stop += 1
        frame_logits = logits[batch_index, onset:stop]
        weights = onset_weights[: stop - onset]
        losses.append((F.softplus(-frame_logits) * weights).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def differentiable_event_support(
    logits: torch.Tensor,
    config: EventObjectiveConfig,
    valid_mask: torch.Tensor | None = None,
    reset_before: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate the 3-hop/180-of-360ms/80ms-tail production evidence."""

    probability = torch.sigmoid(logits)
    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    if reset_before is None:
        reset_before = torch.zeros_like(logits, dtype=torch.bool)
    support = torch.zeros_like(probability)
    required_fraction = config.required_evidence_hops / config.evidence_window_hops
    valid_values = valid_mask.detach().cpu().tolist()
    reset_values = reset_before.detach().cpu().tolist()
    for batch_index in range(logits.shape[0]):
        for start, stop in _valid_segments_from_values(
            valid_values[batch_index],
            reset_values[batch_index],
        ):
            segment = probability[batch_index : batch_index + 1, start:stop]
            smoothed = _causal_mean(segment, config.smoothing_hops)
            accepted = torch.sigmoid(
                (smoothed - config.decision_threshold) / config.event_temperature
            )
            window_fraction = _rolling_mean(
                accepted,
                config.evidence_window_hops,
            )
            duration_support = torch.sigmoid(
                (window_fraction - required_fraction) / config.event_temperature
            )
            tail_fraction = _rolling_mean(accepted, config.tail_hops)
            tail_support = torch.sigmoid(
                (tail_fraction - 0.5) / config.event_temperature
            )
            support[batch_index, start:stop] = (
                duration_support * tail_support * accepted
            ).squeeze(0)
    return support


def event_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    reset_before: torch.Tensor,
    far_only_mask: torch.Tensor,
    config: EventObjectiveConfig,
) -> torch.Tensor:
    support = differentiable_event_support(
        logits,
        config,
        valid_mask,
        reset_before,
    )
    losses: list[torch.Tensor] = []
    active = (target >= 0.5) & valid_mask
    active_values = active.detach().cpu().tolist()
    reset_values = reset_before.detach().cpu().tolist()
    far_only_values = far_only_mask.detach().cpu().tolist()
    previous = torch.zeros_like(active)
    previous[:, 1:] = active[:, :-1]
    previous[reset_before] = False
    onset_locations = torch.nonzero(active & ~previous, as_tuple=False).tolist()
    for index in range(target.shape[0]):
        if far_only_values[index]:
            values = support[index][valid_mask[index]]
            if values.numel():
                peak = _soft_max(values, config.event_temperature)
                losses.append(-torch.log1p(-peak.clamp(max=1.0 - 1e-6)))
            continue

        for onset_batch, onset in onset_locations:
            if onset_batch != index:
                continue
            stop = onset
            while (
                stop < target.shape[1]
                and active_values[index][stop]
                and (stop == onset or not reset_values[index][stop])
            ):
                stop += 1
            deadline = min(stop, onset + config.evidence_window_hops * 2)
            values = support[index, onset:deadline]
            if values.numel():
                peak = _soft_max(values, config.event_temperature)
                losses.append(-torch.log(peak.clamp_min(1e-6)))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def reset_cold_start_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    reset_before: torch.Tensor,
    positive_weight: float,
    config: EventObjectiveConfig,
) -> torch.Tensor:
    """Give extra weight to logits already recomputed from a cleared state.

    ``apply_stream_resets`` supplies the zero-state logits. Gap and padding
    frames remain excluded by ``valid_mask``.
    """

    losses: list[torch.Tensor] = []
    reset_locations = torch.nonzero(
        reset_before[:, 1:] & valid_mask[:, 1:],
        as_tuple=False,
    )
    for batch_index, reset_offset in reset_locations.tolist():
        reset_frame = reset_offset + 1
        stop_frame = min(
            target.shape[1],
            reset_frame + config.reset_warmup_hops,
        )
        reset_target = target[
            batch_index : batch_index + 1,
            reset_frame:stop_frame,
        ]
        reset_valid = valid_mask[
            batch_index : batch_index + 1,
            reset_frame:stop_frame,
        ]
        losses.append(
            weighted_frame_bce(
                logits[batch_index : batch_index + 1, reset_frame:stop_frame],
                reset_target,
                reset_valid,
                positive_weight,
            )
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def training_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    reset_before: torch.Tensor,
    far_only_mask: torch.Tensor,
    positive_weight: float,
    config: EventObjectiveConfig,
) -> dict[str, torch.Tensor]:
    config.validate()
    zero = logits.sum() * 0.0
    components = {
        "bce": weighted_frame_bce(
            logits,
            target,
            valid_mask,
            positive_weight,
        ),
        "hard_negative": (
            hard_negative_loss(
                logits,
                target,
                valid_mask,
                reset_before,
                far_only_mask,
                config,
            )
            if config.hard_negative_weight > 0.0
            else zero
        ),
        "onset": (
            onset_loss(logits, target, valid_mask, reset_before, config)
            if config.onset_weight > 0.0
            else zero
        ),
        "event": (
            event_loss(
                logits,
                target,
                valid_mask,
                reset_before,
                far_only_mask,
                config,
            )
            if config.event_weight > 0.0
            else zero
        ),
        "reset": (
            reset_cold_start_loss(
                logits,
                target,
                valid_mask,
                reset_before,
                positive_weight,
                config,
            )
            if config.reset_weight > 0.0
            else zero
        ),
    }
    components["total"] = (
        components["bce"]
        + config.hard_negative_weight * components["hard_negative"]
        + config.onset_weight * components["onset"]
        + config.event_weight * components["event"]
        + config.reset_weight * components["reset"]
    )
    return components
