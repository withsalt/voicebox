"""Regenerate numerical test references with NumPy (optional developer tool).

The reference uses NumPy's independent real FFT and dense mel-filter matrix.
No models or audio downloads are needed. Run from any working directory.
"""
import json
from pathlib import Path

import numpy as np

count = 4096
index = np.arange(count)
samples = (0.15 * np.sin(2 * np.pi * 431 * index / 16000)
           + 0.07 * np.cos(2 * np.pi * 1277 * index / 16000)).astype(np.float32)
window = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 399)).astype(np.float32)
freq = 1127 * np.log1p(np.arange(256) * 16000 / 512 / 700)
points = np.linspace(1127 * np.log1p(20 / 700), 1127 * np.log1p(8000 / 700), 82)
filters = np.array([
    np.maximum(0, np.minimum((freq - points[i]) / (points[i + 1] - points[i]),
                            (points[i + 2] - freq) / (points[i + 2] - points[i + 1])))
    for i in range(80)
])
frames = []
for start in range(0, count - 399, 160):
    frame = samples[start:start + 400].astype(np.float64) * 32768
    frame -= frame.mean()
    frame[1:] -= 0.97 * frame[:-1].copy()
    frame[0] *= 0.03
    power = np.abs(np.fft.rfft(frame * window, n=512)[:256]) ** 2
    frames.append(np.log(np.maximum(filters @ power, 1.1920929e-7)).astype(np.float32))
shift = (np.arange(400) % 7 - 3).astype(np.float32)
scale = (0.1 + (np.arange(400) % 11) * 0.02).astype(np.float32)
lfr = []
for time in range(0, len(frames) - 2, 3):
    stacked = np.concatenate([frames[max(0, time + j)] for j in [-2, -1, 0, 1, 2]])
    lfr.append(((stacked + shift) * scale).tolist())
destination = Path(__file__).resolve().parent.parent / 'tests/golden/frontend.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps({
    'sampleCount': count,
    'firstFbank': frames[0].tolist(),
    'firstLfr': lfr[0],
    'lastLfr': lfr[-1],
    'lfrCount': len(lfr),
}, indent=2) + '\n', encoding='utf-8')
print(destination)
