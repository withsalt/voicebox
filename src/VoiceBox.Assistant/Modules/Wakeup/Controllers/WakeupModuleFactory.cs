using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Core.Services.WakeupService;

namespace VoiceBox.Assistant.Modules.Wakeup.Controllers;

/// <summary>Composition adapter that creates a wakeup module without exposing its provider type.</summary>
internal sealed class WakeupModuleFactory : IWakeupModuleFactory
{
    private readonly IWakeupService _wakeup;
    private readonly AssistantAudioFormat _audioFormat;
    private readonly VoiceAssistantOptions _options;

    public WakeupModuleFactory(
        IWakeupService wakeup,
        AssistantAudioFormat audioFormat,
        VoiceAssistantOptions options)
    {
        _wakeup = wakeup ?? throw new ArgumentNullException(nameof(wakeup));
        _audioFormat = audioFormat ?? throw new ArgumentNullException(nameof(audioFormat));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public IWakeupModule Create(
        ISpeakerEvidenceProvider? speakerEvidence,
        TimeProvider timeProvider,
        ILogger? logger = null)
    {
        ArgumentNullException.ThrowIfNull(timeProvider);
        return new WakeupModule(
            _wakeup,
            ComputePreRollSamples(),
            _options.Conversation.ContinuousConversationIdleTimeout,
            _audioFormat.WakeupSampleRate,
            _audioFormat.WakeupChannels,
            _options.Wakeup.ReplayLeadInMs,
            speakerEvidence,
            timeProvider,
            _options.Speaker.WakeupStrongScoreMargin,
            _audioFormat.SamplesFromMillisecondsToInt32(
                _options.Speaker.WakeupPostTriggerEvidenceMs,
                _audioFormat.WakeupSampleRate,
                _audioFormat.WakeupChannels),
            logger);
    }

    private int ComputePreRollSamples()
    {
        if (_options.Wakeup.PreRollMs <= 0)
            return 0;

        return _audioFormat.SamplesFromMillisecondsToInt32(
            _options.Wakeup.PreRollMs,
            _audioFormat.WakeupSampleRate,
            _audioFormat.WakeupChannels);
    }
}
