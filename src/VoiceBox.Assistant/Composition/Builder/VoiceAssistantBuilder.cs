using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text.Json;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Composition.Configuration;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;

namespace VoiceBox.Assistant.Composition.Builder;

/// <summary>
/// Typed, one-shot composition root for a VoiceAssistant instance. Registrations are immutable
/// after <see cref="Build"/>; it deliberately accepts no service provider or manager container.
/// </summary>
public sealed class VoiceAssistantBuilder
{
    private IAudioDuplexService? _duplexAudio;
    private SplitAudioRegistration? _splitAudio;
    private IVoiceActivityDetectService? _turnBoundary;
    private IASRService? _recognition;
    private IConversationSession? _conversation;
    private IStreamTTSService? _speechSynthesis;
    private IWakeupService? _wakeup;
    private ISpeakerService? _speaker;
    private IAcousticProcessingMetricsSource? _acousticDiagnostics;
    private IAcousticPathDelayEstimatorFactory? _acousticPathDelayEstimatorFactory;
    private VoiceAssistantOptions _options = new();
    private bool _optionsSpecified;
    private bool _bargeInSpecified;
    private bool _acousticDiagnosticsSpecified;

    internal VoiceAssistantBuilder()
    {
    }

    public VoiceAssistantBuilder UseAudio(IAudioDuplexService audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        EnsureAudioUnset();
        _duplexAudio = audio;
        return this;
    }

    public VoiceAssistantBuilder UseAudio(IAudioRecordService record, IAudioPlayService playback)
    {
        ArgumentNullException.ThrowIfNull(record);
        ArgumentNullException.ThrowIfNull(playback);
        EnsureAudioUnset();

        if (ReferenceEquals(record, playback) && record is IAudioDuplexService duplex)
        {
            _duplexAudio = duplex;
        }
        else
        {
            _splitAudio = new SplitAudioRegistration(record, playback);
        }

        return this;
    }

    public VoiceAssistantBuilder UseTurnBoundary(IVoiceActivityDetectService turnBoundary)
    {
        _turnBoundary = SetOnce(_turnBoundary, turnBoundary, "TurnBoundary");
        return this;
    }

    public VoiceAssistantBuilder UseSpeechRecognition(IASRService recognition)
    {
        _recognition = SetOnce(_recognition, recognition, "SpeechRecognition");
        return this;
    }

    /// <summary>
    /// 挂载调用方的业务会话。Assistant 借用该实例, 不转移所有权, 也不会调用它的 DisposeAsync;
    /// 如果它实现 <see cref="IConversationLifecycle"/>, 运行期会调用 StartAsync/StopAsync。
    /// </summary>
    public VoiceAssistantBuilder UseConversation(IConversationSession conversation)
    {
        _conversation = SetOnce(_conversation, conversation, "Conversation");
        return this;
    }

    public VoiceAssistantBuilder UseSpeechSynthesis(IStreamTTSService speechSynthesis)
    {
        _speechSynthesis = SetOnce(_speechSynthesis, speechSynthesis, "SpeechSynthesis");
        return this;
    }

    public VoiceAssistantBuilder AddWakeup(IWakeupService wakeup)
    {
        _wakeup = SetOnce(_wakeup, wakeup, "WakeupGate");
        return this;
    }

    /// <summary>
    /// Installs the owner-speaker provider. Online verification only becomes effective when KWS is
    /// enabled through <see cref="AddWakeup"/>; without KWS the provider is never consulted at
    /// runtime (standalone enrollment may still use it).
    /// </summary>
    public VoiceAssistantBuilder AddSpeakerVerification(ISpeakerService speaker)
    {
        _speaker = SetOnce(_speaker, speaker, "SpeakerEvidence");
        return this;
    }

    /// <summary>
    /// Installs the optional acoustic-processing diagnostics source. The audio frontend remains
    /// responsible for running the processor; this registration only exposes its typed metrics
    /// port to the assistant runtime.
    /// </summary>
    public VoiceAssistantBuilder AddAcousticDiagnostics(IAcousticProcessingMetricsSource diagnostics)
    {
        _acousticDiagnostics = SetOnce(
            _acousticDiagnostics,
            diagnostics,
            "AcousticProcessingDiagnostics");
        _acousticDiagnosticsSpecified = true;
        return this;
    }

    /// <summary>
    /// 安装播放期打断能力及其流级声学路径延迟估计器工厂。
    /// </summary>
    public VoiceAssistantBuilder AddBargeIn(VoiceAssistantBargeInOptions options, IAcousticPathDelayEstimatorFactory delayEstimatorFactory)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(delayEstimatorFactory);
        if (_bargeInSpecified)
        {
            throw new AssistantCompositionException("VBX-COMP-002", "BargeIn has already been registered for this builder.");
        }

        _options = CloneOptions(_options);
        _options.BargeIn = CloneBargeInOptions(options);
        _options.BargeIn.Enable = true;
        _acousticPathDelayEstimatorFactory = delayEstimatorFactory;
        _bargeInSpecified = true;
        return this;
    }

    public VoiceAssistantBuilder UseOptions(VoiceAssistantOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (_optionsSpecified || _wakeup is not null || _speaker is not null ||
            _bargeInSpecified || _acousticDiagnosticsSpecified)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-002",
                "VoiceAssistantOptions has already been registered for this builder.");
        }

        _options = CloneOptions(options);
        _optionsSpecified = true;
        return this;
    }

    public VoiceAssistant Build()
        => BuildWithDiagnostics().Assistant;

    /// <summary>
    /// Builds an assistant together with its sanitized composition identity for hosting diagnostics.
    /// </summary>
    public (
        VoiceAssistant Assistant,
        string ProfileId,
        string CapabilityGraphHash,
        IReadOnlyList<string> Capabilities) BuildWithDiagnostics(ILogger<VoiceAssistant>? logger = null)
    {
        AssistantComposition composition = BuildComposition(out VoiceAssistantOptions options);
        var assistant = new VoiceAssistant(composition, Options.Create(options), logger);
        return (
            assistant,
            composition.Snapshot.ProfileId,
            composition.Snapshot.CapabilityGraphHash,
            composition.Snapshot.Capabilities);
    }

    /// <summary>
    /// Builds an assistant with deterministic clock and benchmark hooks while retaining the same
    /// typed composition path as <see cref="Build"/>. This is intentionally internal: production
    /// callers should use <see cref="Build"/> and the normal system clock.
    /// </summary>
    internal VoiceAssistant BuildForBenchmark(
        TimeProvider timeProvider,
        VoiceAssistantBenchmarkProbe benchmarkProbe,
        ILogger<VoiceAssistant>? logger = null,
        IPlaybackBargeInDetectorFactory? playbackBargeInDetectorFactory = null,
        Modules.Speaker.Contracts.ISpeakerModuleFactory? speakerModuleFactory = null)
    {
        ArgumentNullException.ThrowIfNull(timeProvider);
        ArgumentNullException.ThrowIfNull(benchmarkProbe);
        AssistantComposition composition = BuildComposition(out VoiceAssistantOptions options);
        return new VoiceAssistant(
            composition,
            Options.Create(options),
            timeProvider,
            benchmarkProbe,
            logger,
            playbackBargeInDetectorFactory,
            speakerModuleFactory);
    }

    /// <summary>Validates an options object with the same oracle used by every composition path.</summary>
    public static void ValidateOptions(VoiceAssistantOptions options)
        => VoiceAssistantOptionsValidator.Validate(options, nameof(options));

    internal AssistantComposition BuildComposition()
        => BuildComposition(out _);

    private AssistantComposition BuildComposition(out VoiceAssistantOptions options)
    {
        IAssistantAudioFrontend audio = CreateAudioFrontend();
        try
        {
            return BuildComposition(audio, out options);
        }
        catch
        {
            // The split adapter subscribes to a borrowed recorder in its constructor. A
            // rejected composition must detach it because no Assistant will own its disposal.
            (audio as SplitAssistantAudioFrontend)?.Dispose();
            throw;
        }
    }

    private AssistantComposition BuildComposition(IAssistantAudioFrontend audio, out VoiceAssistantOptions options)
    {
        IVoiceActivityDetectService turnBoundary = Require(_turnBoundary, "VBX-COMP-001", "TurnBoundary");
        IASRService recognition = Require(_recognition, "VBX-COMP-001", "SpeechRecognition");
        IConversationSession conversation = Require(_conversation, "VBX-COMP-001", "Conversation");
        IStreamTTSService synthesis = Require(_speechSynthesis, "VBX-COMP-001", "SpeechSynthesis");

        options = CloneOptions(_options);
        options.Wakeup.Enable = _wakeup is not null;
        options.BargeIn.Enable = _bargeInSpecified;

        AssistantCompositionProfile profile = audio is DuplexAssistantAudioFrontend
            ? AssistantCompositionProfile.BasicDuplexQa
            : AssistantCompositionProfile.BasicSplitAudioQa;
        var graph = new AssistantCapabilityGraph(
            profile,
            audio,
            turnBoundary,
            recognition,
            conversation,
            synthesis,
            _wakeup,
            _speaker,
            _acousticDiagnostics,
            options,
            acousticPathDelayEstimatorFactory: _acousticPathDelayEstimatorFactory);
        return graph.ValidateAndCreateComposition();
    }

    private IAssistantAudioFrontend CreateAudioFrontend()
    {
        if (_duplexAudio is not null)
            return new DuplexAssistantAudioFrontend(_duplexAudio, _acousticDiagnostics);
        if (_splitAudio is { } split)
            return new SplitAssistantAudioFrontend(split.Record, split.Playback);

        throw new AssistantCompositionException(
            "VBX-COMP-001",
            "AudioCapture and AudioPlayback must be registered exactly once before Build.");
    }

    private void EnsureAudioUnset()
    {
        if (_duplexAudio is not null || _splitAudio is not null)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-002",
                "Audio endpoint has already been registered for this builder.");
        }
    }

    private static T SetOnce<T>(T? existing, T value, string role)
        where T : class
    {
        ArgumentNullException.ThrowIfNull(value);
        if (existing is not null)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-002",
                $"{role} has already been registered for this builder.");
        }

        return value;
    }

    private static T Require<T>(T? value, string code, string role)
        where T : class
        => value ?? throw new AssistantCompositionException(
            code,
            $"Required role '{role}' is missing before Build.");

    private static VoiceAssistantOptions CloneOptions(VoiceAssistantOptions options) => options.Clone();

    private static VoiceAssistantBargeInOptions CloneBargeInOptions(VoiceAssistantBargeInOptions options)
    {
        // Same delegate constraint as VoiceAssistantOptions.Clone: the shared clone contract skips
        // the SessionOptionsFactory that System.Text.Json cannot round-trip, and it is re-attached
        // by reference so AddBargeIn neither throws nor drops a caller's ONNX session.
        string json = JsonSerializer.Serialize(options, AssistantOptionsCloneSerializer.Json);
        VoiceAssistantBargeInOptions clone = JsonSerializer.Deserialize<VoiceAssistantBargeInOptions>(
            json,
            AssistantOptionsCloneSerializer.Json)
            ?? throw new InvalidOperationException("VoiceAssistantBargeInOptions could not be cloned.");
        if (options.Dtd?.Model?.SessionOptionsFactory is { } sessionOptionsFactory &&
            clone.Dtd?.Model is { } clonedModel)
        {
            clonedModel.SessionOptionsFactory = sessionOptionsFactory;
        }

        return clone;
    }

    private readonly record struct SplitAudioRegistration(IAudioRecordService Record, IAudioPlayService Playback);
}
