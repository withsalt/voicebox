# VoiceBox.DTD.RenderAware

Independent streaming render-aware double-talk detection for VoiceBox.

```csharp
using Microsoft.Extensions.Options;
using VoiceBox.DTD.RenderAware;
using VoiceBox.DTD.RenderAware.Models;

using IRenderAwareDtdService detector = new RenderAwareDtdService(
    Options.Create(new RenderAwareDtdOptions
    {
        Threshold = 0.55f,
    }));

if (detector.Process(microphoneHop, renderHop, 1, 16000) is { } result)
{
    Console.WriteLine(
        $"{result.Timestamp}: raw={result.Probability:F3}, " +
        $"smooth={result.SmoothedProbability:F3}, near-end={result.IsNearEndSpeech}");
}
```

Each service instance owns overlap and recurrent state for one synchronized
microphone/render stream. Public methods are serialized and do not modify input
buffers. Call `Reset` on stream discontinuities. The first 20 ms hop only warms
the 40 ms model window.

The packaged `0.55` decision threshold is a validation-set operating point, not
a universal barge-in setting. Production deployments must calibrate it against
their device, acoustic path, latency alignment, and false-interruption target.
