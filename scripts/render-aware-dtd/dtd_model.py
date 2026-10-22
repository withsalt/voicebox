from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DtdModelConfig:
    sample_rate: int = 16_000
    frame_length: int = 640
    hop_length: int = 320
    frequency_bands: int = 48
    projection_size: int = 192
    hidden_size: int = 192
    gru_layers: int = 2
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DtdComplexity:
    trainable_parameters: int
    frontend_macs_per_second: int
    neural_macs_per_second: int
    total_macs_per_second: int

    def to_dict(self) -> dict[str, int | float]:
        values: dict[str, int | float] = asdict(self)
        values["total_mmacs_per_second"] = self.total_macs_per_second / 1_000_000
        return values


def _hz_to_mel(frequency_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequency_hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


class FixedComplexFilterbank(nn.Module):
    """A deterministic STFT-like front end that exports as ordinary Conv ops."""

    def __init__(self, config: DtdModelConfig):
        super().__init__()
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.frequency_bands = config.frequency_bands

        low_mel = _hz_to_mel(torch.tensor(60.0))
        high_mel = _hz_to_mel(torch.tensor(config.sample_rate / 2.0 - 100.0))
        center_hz = _mel_to_hz(
            torch.linspace(low_mel, high_mel, config.frequency_bands)
        )
        sample_index = torch.arange(config.frame_length, dtype=torch.float32)
        phase = (
            2.0
            * torch.pi
            * center_hz[:, None]
            * sample_index[None, :]
            / config.sample_rate
        )
        window = torch.hann_window(config.frame_length, periodic=False)
        real = torch.cos(phase) * window
        imag = -torch.sin(phase) * window
        real = real / real.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        imag = imag / imag.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        self.register_buffer(
            "filters",
            torch.cat((real, imag), dim=0).unsqueeze(1),
            persistent=True,
        )

    def _complex_spectrum(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The fixed 640-tap filterbank is small. Native CUDA also avoids
        # coupling training to the optional cuDNN convolution sublibraries,
        # while cuDNN remains enabled for the substantially heavier GRU.
        with torch.backends.cudnn.flags(enabled=not waveform.is_cuda):
            spectrum = F.conv1d(
                waveform.unsqueeze(1),
                self.filters,
                stride=self.hop_length,
            )
        return spectrum.split(self.frequency_bands, dim=1)

    def forward(self, microphone: torch.Tensor, render: torch.Tensor) -> torch.Tensor:
        mic_real, mic_imag = self._complex_spectrum(microphone)
        render_real, render_imag = self._complex_spectrum(render)

        epsilon = 1e-6
        mic_power = mic_real.square() + mic_imag.square()
        render_power = render_real.square() + render_imag.square()
        mic_log_power = torch.log(mic_power + epsilon)
        render_log_power = torch.log(render_power + epsilon)
        log_ratio = (mic_log_power - render_log_power).clamp(-12.0, 12.0)

        normalizer = torch.sqrt((mic_power + epsilon) * (render_power + epsilon))
        phase_cosine = (
            mic_real * render_real + mic_imag * render_imag
        ) / normalizer
        phase_sine = (
            mic_imag * render_real - mic_real * render_imag
        ) / normalizer

        features = torch.cat(
            (
                mic_log_power.clamp(-14.0, 6.0) / 7.0,
                render_log_power.clamp(-14.0, 6.0) / 7.0,
                log_ratio / 6.0,
                phase_cosine,
                phase_sine,
            ),
            dim=1,
        )
        return features.transpose(1, 2)


class RenderAwareDtd(nn.Module):
    def __init__(self, config: DtdModelConfig | None = None):
        super().__init__()
        self.config = config or DtdModelConfig()
        self.frontend = FixedComplexFilterbank(self.config)
        feature_size = self.config.frequency_bands * 5
        self.input_projection = nn.Sequential(
            nn.Linear(feature_size, self.config.projection_size),
            nn.LayerNorm(self.config.projection_size),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
        )
        self.temporal = nn.GRU(
            input_size=self.config.projection_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.gru_layers,
            dropout=self.config.dropout if self.config.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, 1),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        reference = next(self.parameters())
        return torch.zeros(
            self.config.gru_layers,
            batch_size,
            self.config.hidden_size,
            device=device or reference.device,
            dtype=dtype or reference.dtype,
        )

    def forward(
        self,
        microphone: torch.Tensor,
        render: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if microphone.ndim != 2 or render.ndim != 2:
            raise ValueError("microphone and render must have shape [batch, samples]")
        if microphone.shape != render.shape:
            raise ValueError("microphone and render must have identical shapes")
        if microphone.shape[1] < self.config.frame_length:
            raise ValueError(
                f"audio must contain at least {self.config.frame_length} samples"
            )

        features = self.frontend(microphone, render)
        projected = self.input_projection(features)
        encoded, next_state = self.temporal(projected, state)
        logits = self.classifier(encoded).squeeze(-1)
        return logits, next_state

    def complexity(self) -> DtdComplexity:
        """Return deterministic streaming MAC and parameter counts."""
        config = self.config
        frames_per_second = config.sample_rate // config.hop_length
        filter_count = config.frequency_bands * 2
        frontend_per_frame = (
            2 * filter_count * config.frame_length
        )
        projection_per_frame = (
            config.frequency_bands * 5 * config.projection_size
        )
        recurrent_per_frame = 0
        input_size = config.projection_size
        for _ in range(config.gru_layers):
            recurrent_per_frame += (
                3 * config.hidden_size * (input_size + config.hidden_size)
            )
            input_size = config.hidden_size
        classifier_per_frame = (
            config.hidden_size * (config.hidden_size // 2)
            + config.hidden_size // 2
        )
        frontend = frontend_per_frame * frames_per_second
        neural = (
            projection_per_frame
            + recurrent_per_frame
            + classifier_per_frame
        ) * frames_per_second
        return DtdComplexity(
            trainable_parameters=sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            frontend_macs_per_second=frontend,
            neural_macs_per_second=neural,
            total_macs_per_second=frontend + neural,
        )


class StreamingDtdWrapper(nn.Module):
    def __init__(self, model: RenderAwareDtd):
        super().__init__()
        self.model = model

    def forward(
        self,
        microphone: torch.Tensor,
        render: torch.Tensor,
        state_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, state_out = self.model(microphone, render, state_in)
        return torch.sigmoid(logits).unsqueeze(-1), state_out

