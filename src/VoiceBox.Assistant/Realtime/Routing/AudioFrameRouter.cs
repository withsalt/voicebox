using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Realtime.Routing;

/// <summary>
/// Static, typed data-plane fan-out. Frames never enter the control ingress or Kernel; callers
/// provide the sinks at Build time and the router performs only synchronous ordered dispatch.
/// A duplex frame is first sliced onto the identity tap for the speaker sinks and then handed, in
/// full, to the render-aware sinks. Neither consumer runs model inference inside the dispatch.
/// </summary>
internal sealed class AudioFrameRouter
{
    private readonly IProcessedCaptureFrameSink[] _processedCapture;
    private readonly IRenderReferenceFrameSink[] _renderReference;
    private readonly ISpeakerCaptureFrameSink[] _speakerCapture;
    private readonly SpeakerCaptureFrameFactory? _speakerFrames;

    public AudioFrameRouter(
        IReadOnlyList<IProcessedCaptureFrameSink> processedCapture,
        IReadOnlyList<IRenderReferenceFrameSink>? renderReference = null,
        IReadOnlyList<ISpeakerCaptureFrameSink>? speakerCapture = null,
        SpeakerCaptureFrameFactory? speakerFrames = null)
    {
        ArgumentNullException.ThrowIfNull(processedCapture);
        _processedCapture = processedCapture.ToArray();
        _renderReference = renderReference is null ? [] : renderReference.ToArray();
        _speakerCapture = speakerCapture is null ? [] : speakerCapture.ToArray();
        if (_speakerCapture.Length > 0 && speakerFrames is null)
        {
            throw new ArgumentException(
                "Speaker capture sinks require the identity tap frame factory.",
                nameof(speakerFrames));
        }
        _speakerFrames = speakerFrames;
    }

    public bool HasDuplexSinks => _renderReference.Length > 0 || _speakerCapture.Length > 0;

    public void RouteProcessedCapture(in ProcessedCaptureFrame frame)
    {
        IProcessedCaptureFrameSink[] sinks = _processedCapture;
        for (int i = 0; i < sinks.Length; i++)
            sinks[i].OnProcessedCapture(in frame);
    }

    public void RouteDuplexFrame(in DuplexAudioFrame frame)
    {
        ISpeakerCaptureFrameSink[] speakerSinks = _speakerCapture;
        if (speakerSinks.Length > 0 &&
            _speakerFrames!.TryCreate(in frame, out SpeakerCaptureFrame identity))
        {
            for (int i = 0; i < speakerSinks.Length; i++)
                speakerSinks[i].OnSpeakerCapture(in identity);
        }

        IRenderReferenceFrameSink[] sinks = _renderReference;
        for (int i = 0; i < sinks.Length; i++)
            sinks[i].OnRenderReference(in frame);
    }
}
