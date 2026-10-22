# VoiceBox.Speaker.WeSpeaker

Production-oriented `ISpeakerService` provider for WeSpeaker's non-LM
`cnceleb_resnet34.onnx`.

## Model Contract

- 16 kHz mono `float32` audio in `[-1,1]`
- 80-bin Kaldi FBank, 25 ms frame / 10 ms shift, Hamming window, `dither=0`
- waveform scale `32768`, per-utterance or per-window CMN, no CVN
- ONNX input `feats [B,T,80]`, output `embs [B,256]`
- L2-normalized embeddings and raw cosine scores in `[-1,1]`

The model has global statistics pooling and no cache tensors. `FeedAudio` is
application-level streaming: it incrementally computes FBank frames, then runs the
segment model over a sliding window.

## Registration

```csharp
using Microsoft.Extensions.Options;
using VoiceBox.Speaker.WeSpeaker;
using VoiceBox.Speaker.WeSpeaker.Models;

using var speaker = new WeSpeakerService(
    Options.Create(new WeSpeakerOptions
    {
        WindowMs = 2000,
        HopMs = 250,
        Threshold = 0.55f,
    }));

speaker.Enroll("alice", enrollmentAudio);
float score = speaker.Verify(enrollmentAudio, trialAudio);
speaker.OnResult += (_, result) =>
    Console.WriteLine($"{result.AudioTime}: {result.BestSpeaker} {result.BestScore:F3}");
speaker.FeedAudio(streamingAudio);
```

The packaged model SHA-256 is checked at service startup. A consuming executable
must provide a native ONNX Runtime package such as `Microsoft.ML.OnnxRuntime`;
the library intentionally references only `Microsoft.ML.OnnxRuntime.Managed`.

`0.55` is a small-corpus smoke-test baseline, not an upstream operating point.
Before production deployment, calibrate `Threshold`, window length, microphone
distance, noise conditions, and enrollment protocol on representative data.
Review `runtimes/models/speaker/wespeaker/MODEL.md` before redistributing weights.
