# Render-aware DTD model

This package contains the VoiceBox render-aware near-end double-talk detector
trained by `scripts/render-aware-dtd`. The packaged model is v6 candidate-01 at
validation-frozen epoch 147 (`epoch-147.pt`). This checkpoint was selected by
the validation-only production-event Pareto policy before the fixed test and
VoiceBox AEC Challenge benchmark comparison.

This integration is explicit and requested. The fixed benchmark report remains
`NOT ACCEPTED / INCOMPLETE`; candidate-16 non-regression gates were not met.

## Runtime contract

- 16 kHz mono float32 microphone and synchronized render audio
- 320-sample (20 ms) streaming hop
- 640-sample (40 ms) two-hop analysis window
- 80 frequency bands, projection 288, GRU hidden 288 with 3 layers
- 1,656,577 trainable parameters and 92.7304 MMAC/s
- GRU state `[3,1,288]`, initialized to zero and carried between windows
- `nearend_probability [1,1,1]` in `[0,1]`

The exported `0.55` threshold is the validation selection operating point.
It is not a universal production barge-in threshold. Calibrate the
threshold, microphone path, render gain, and acoustic delay
using representative devices and rooms.

The library references ONNX Runtime managed APIs only. The consuming executable
must provide a native runtime package such as `Microsoft.ML.OnnxRuntime`.
