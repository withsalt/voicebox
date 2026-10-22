using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Controllers;

/// <summary>
/// Composition adapter that hides the speaker provider behind the module factory port. The
/// effective purposes are computed by the capability graph from the KWS gate; the factory never
/// re-derives them from provider presence.
/// </summary>
internal sealed class SpeakerModuleFactory : ISpeakerModuleFactory
{
    private readonly ISpeakerService _speaker;
    private readonly VoiceAssistantSpeakerOptions _options;
    private readonly SpeakerPurposeSet _effectivePurposes;

    public SpeakerModuleFactory(
        ISpeakerService speaker,
        VoiceAssistantSpeakerOptions options,
        SpeakerPurposeSet effectivePurposes)
    {
        _speaker = speaker ?? throw new ArgumentNullException(nameof(speaker));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (!effectivePurposes.Any)
            throw new ArgumentException("At least one effective online purpose is required.", nameof(effectivePurposes));
        _effectivePurposes = effectivePurposes;
    }

    public SpeakerPurposeSet EffectivePurposes => _effectivePurposes;

    public ISpeakerModule Create(
        SpeakerCaptureDescriptor descriptor,
        int audioSampleRate,
        int audioChannels,
        ILogger logger)
        => new SpeakerModule(
            _speaker,
            _options,
            _effectivePurposes,
            descriptor,
            audioSampleRate,
            audioChannels,
            logger);
}
