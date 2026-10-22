# WeSpeaker ONNX Validation

This directory generates golden vectors for WeSpeaker ONNX validation.

The reference implementation uses the exact preprocessing from
[`wespeaker/bin/infer_onnx.py`](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/infer_onnx.py):
16 kHz mono audio, `32768` waveform scaling, 80-bin Hamming Kaldi FBank,
25/10 ms framing, `dither=0`, utterance CMN, and `feats`/`embs` ONNX tensors.

Run from the repository root:

```powershell
conda run -n voicebox python scripts/wespeaker-onnx-validation/generate_vectors.py
```

The three small speech files come from SpeechBrain's Apache-2.0 licensed test
assets, pinned by source revision and individual SHA-256 values in the script.
The license text is included as `SPEECHBRAIN_LICENSE.txt`.
Generated ONNX embeddings are model-derived test fixtures. Review the model
terms in `src/VoiceBox.Speaker.WeSpeaker/runtimes/models/speaker/wespeaker/MODEL.md`.
