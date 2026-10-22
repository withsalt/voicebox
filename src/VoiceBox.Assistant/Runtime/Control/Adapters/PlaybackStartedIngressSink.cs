using System.Diagnostics;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Adapters;

/// <summary>Admits the first successful playback write as a low-frequency control fact.</summary>
internal sealed class PlaybackStartedIngressSink : IPlaybackStartedSink
{
    private readonly AssistantControlIngress _ingress;

    public PlaybackStartedIngressSink(AssistantControlIngress ingress)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
    }

    public void OnPlaybackStarted(long turnId)
    {
        if (_ingress.TryEnqueueCritical(
                AssistantSignal.PlaybackTransportStarted(turnId),
                Stopwatch.GetTimestamp(),
                AssistantSignalSource.Playback,
                out _))
        {
            return;
        }

        if (!_ingress.HasTerminalFault)
        {
            _ingress.TryTerminate(
                new AssistantFailure(
                    "VBX-CTRL-021",
                    "capacity",
                    "playback-data-plane",
                    "playback-start-admission-failed"),
                Stopwatch.GetTimestamp(),
                out _);
        }
    }
}
