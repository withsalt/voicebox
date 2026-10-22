from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioMetadata:
    path: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def inspect_audio(path: str | Path) -> AudioMetadata:
    """Read the file header and return auditable source-format metadata."""
    resolved = Path(path).resolve()
    try:
        information = sf.info(resolved)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot inspect audio file: {resolved}") from error
    if information.samplerate <= 0:
        raise ValueError(f"audio has an invalid sample rate: {resolved}")
    if information.channels <= 0:
        raise ValueError(f"audio has no channels: {resolved}")
    if information.frames <= 0:
        raise ValueError(f"audio is empty: {resolved}")
    return AudioMetadata(
        path=str(resolved),
        sample_rate=int(information.samplerate),
        channels=int(information.channels),
        frames=int(information.frames),
        duration_seconds=float(information.duration),
    )


def read_mono_resampled(
    path: str | Path,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """Read one file using its own header rate and return finite mono PCM."""
    resolved = Path(path)
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")
    try:
        samples, source_sample_rate = sf.read(
            resolved,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"cannot read audio file: {resolved}") from error
    if source_sample_rate <= 0:
        raise ValueError(f"audio has an invalid sample rate: {resolved}")
    if samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(f"audio is empty: {resolved}")

    mono = samples.mean(axis=1, dtype=np.float32)
    if source_sample_rate != target_sample_rate:
        divisor = math.gcd(int(source_sample_rate), target_sample_rate)
        mono = resample_poly(
            mono,
            target_sample_rate // divisor,
            int(source_sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    return np.nan_to_num(
        mono,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).astype(np.float32, copy=False)
