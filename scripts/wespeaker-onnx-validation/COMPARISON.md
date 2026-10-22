# Validation and Upstream Comparison

Validation date: 2026-07-30.

## Numerical Parity

The fixtures were generated with WeSpeaker's official inference preprocessing and
the packaged `cnceleb_resnet34.onnx`. The C# tests compare every stage rather than
only the final decision:

| Stage | Result |
|---|---:|
| Hamming Kaldi FBank vs torchaudio | max error `9.68e-4`, mean error `<9.6e-6` |
| ONNX embedding from identical CMN features | cosine `1.00000000` |
| WAV-to-embedding end to end | cosine `1.00000000` |
| Sliding-window embeddings | worst cosine `1.00000000` |
| Pairwise raw cosine scores | difference below `1e-4` |

The open SpeechBrain smoke corpus produced:

| Pair | Relationship | Python | C# |
|---|---|---:|---:|
| `spk1_snt1` / `spk1_snt2` | same speaker | `0.7662` | `0.7662` |
| `spk1_snt1` / `spk2_snt1` | different speakers | `0.4418` | `0.4418` |
| `spk1_snt2` / `spk2_snt1` | different speakers | `0.3864` | `0.3864` |

This is a functional smoke test, not threshold calibration or an accuracy claim.

## Accuracy Context

The upstream CN-Celeb v2 recipe reports `7.134%` EER and `0.408` minDCF for the
non-LM ResNet34 without AS-Norm or QMF. Those corpus metrics cannot be reproduced
from the three redistribution-friendly test files in this repository. The `_LM`
model reports a lower EER but is explicitly intended for long audio and is not used.

## Performance

Release-mode tests on the development machine measured:

- 3-second offline extraction: approximately `22-23 ms/call`
- 30-second, 2-second-window / 250-ms-hop stream: RTF approximately `0.054`
- streaming allocation: about `1.5 KB/window`

These numbers are regression guards, not cross-machine benchmarks. WeSpeaker's
published one-thread runtime table reports ResNet34 RTF `0.060735` and CAM++ RTF
`0.022978`; ResNet34 is expected to consume more CPU than CAM++.
