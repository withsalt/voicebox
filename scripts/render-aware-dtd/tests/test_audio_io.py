from __future__ import annotations

import numpy as np
import soundfile as sf

from audio_io import inspect_audio, read_mono_resampled


def test_reads_each_file_rate_and_resamples_stereo(tmp_path):
    path = tmp_path / "stereo-48k.wav"
    time = np.arange(4_800, dtype=np.float32) / 48_000
    stereo = np.stack(
        (
            np.sin(2.0 * np.pi * 440.0 * time),
            np.sin(2.0 * np.pi * 660.0 * time),
        ),
        axis=1,
    )
    sf.write(path, stereo, 48_000, subtype="FLOAT")

    metadata = inspect_audio(path)
    samples = read_mono_resampled(path)

    assert metadata.sample_rate == 48_000
    assert metadata.channels == 2
    assert len(samples) == 1_600
    assert samples.dtype == np.float32
    assert np.isfinite(samples).all()


def test_empty_audio_is_rejected(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.empty(0, dtype=np.float32), 16_000)

    try:
        inspect_audio(path)
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty audio should be rejected")
