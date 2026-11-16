using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Runtime.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>Explicit near-end composition inputs; every capability is a typed port.</summary>
internal sealed class NearEndModuleDependencies
{
    public required VoiceAssistantBargeInOptions Options { get; init; }

    public required IActiveNearEndConversationView Conversation { get; init; }

    public required IActiveNearEndDecisionSource Decisions { get; init; }

    public required AssistantAudioFormat AudioFormat { get; init; }

    public required ILogger Logger { get; init; }

    public required IRuntimeLivenessView Runtime { get; init; }

    public required IActiveNearEndPlaybackPort Playback { get; init; }

    public required IActiveNearEndSignalSink Control { get; init; }

    public required INearEndRecognitionAudioSink RecognitionAudio { get; init; }

    /// <summary>Non-null exactly when online identity confirmation is enabled.</summary>
    public ISpeakerEvidenceProvider? SpeakerEvidence { get; init; }

    public TimeProvider? TimeProvider { get; init; }
}
