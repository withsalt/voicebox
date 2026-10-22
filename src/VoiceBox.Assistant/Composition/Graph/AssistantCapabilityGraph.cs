using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using System.Security.Cryptography;
using System.Text;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Composition.Wiring;
using VoiceBox.Assistant.Diagnostics.Metrics;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware;
using VoiceBox.Assistant.Modules.Playback;
using VoiceBox.Assistant.Modules.Recognition.Adapters;
using VoiceBox.Assistant.Modules.Response.Adapters;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.Speaker;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary;
using VoiceBox.Assistant.Modules.Wakeup;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Assistant.Composition.Configuration;
using VoiceBox.Assistant.Modules.TurnBoundary.Adapters;
using VoiceBox.Assistant.Modules.Wakeup.Controllers;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Controllers;
using VoiceBox.Assistant.Modules.Playback.Adapters;

namespace VoiceBox.Assistant.Composition.Graph;

/// <summary>
/// Build-time graph only. The validator turns this into immutable plans; this type is never held
/// by the Runtime and exposes no capability lookup API.
/// </summary>
internal sealed class AssistantCapabilityGraph
{
    private readonly AssistantCompositionProfile _profile;
    private readonly IAssistantAudioFrontend _audio;
    private readonly IVoiceActivityDetectService _turnBoundary;
    private readonly IASRService _recognition;
    private readonly IStreamLLMService _responseGeneration;
    private readonly IStreamTTSService _speechSynthesis;
    private readonly IWakeupService? _wakeup;
    private readonly ISpeakerService? _speaker;
    private readonly IAcousticProcessingMetricsSource? _acousticDiagnostics;
    private readonly IAcousticPathDelayEstimatorFactory? _acousticPathDelayEstimatorFactory;
    private readonly IAssistantToolRuntimeFactory? _toolRuntimeFactory;
    private readonly VoiceAssistantOptions _options;
    private readonly IReadOnlySet<AssistantCapability> _requestedCapabilities;
    private readonly ILogger _detectorLogger;
    private readonly TimeProvider? _timeProvider;

    public AssistantCapabilityGraph(
        AssistantCompositionProfile profile,
        IAssistantAudioFrontend audio,
        IVoiceActivityDetectService turnBoundary,
        IASRService recognition,
        IStreamLLMService responseGeneration,
        IStreamTTSService speechSynthesis,
        IWakeupService? wakeup,
        ISpeakerService? speaker,
        IAcousticProcessingMetricsSource? acousticDiagnostics,
        VoiceAssistantOptions options,
        IAssistantToolRuntimeFactory? toolRuntimeFactory = null,
        IReadOnlySet<AssistantCapability>? requestedCapabilities = null,
        ILogger? detectorLogger = null,
        TimeProvider? timeProvider = null,
        IAcousticPathDelayEstimatorFactory? acousticPathDelayEstimatorFactory = null)
    {
        _requestedCapabilities = requestedCapabilities ?? new HashSet<AssistantCapability>();
        _detectorLogger = detectorLogger ?? NullLogger.Instance;
        _timeProvider = timeProvider;
        _profile = profile;
        _audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _turnBoundary = turnBoundary ?? throw new ArgumentNullException(nameof(turnBoundary));
        _recognition = recognition ?? throw new ArgumentNullException(nameof(recognition));
        _responseGeneration = responseGeneration ?? throw new ArgumentNullException(nameof(responseGeneration));
        _speechSynthesis = speechSynthesis ?? throw new ArgumentNullException(nameof(speechSynthesis));
        _wakeup = wakeup;
        _speaker = speaker;
        _acousticDiagnostics = acousticDiagnostics;
        _acousticPathDelayEstimatorFactory = acousticPathDelayEstimatorFactory;
        _toolRuntimeFactory = toolRuntimeFactory;
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Online identity is gated by <see cref="SpeakerConfirmationOptions.Enable"/>. When it is off,
    /// no purpose is effective. When it is on, each installed scene (KWS, barge-in, continuous
    /// listening) uses the shared confirmation policy.
    /// </summary>
    internal SpeakerPurposeSet EffectiveSpeakerPurposes
        => _options.Speaker.Confirmation?.Enable == true
            ? new SpeakerPurposeSet(
                Wakeup: _options.Wakeup.Enable,
                BargeInCommand: _options.BargeIn.Enable,
                SingleTalk: true)
            : SpeakerPurposeSet.None;

    private bool OnlineSpeakerInstalled => EffectiveSpeakerPurposes.Any;

    public AssistantComposition ValidateAndCreateComposition()
    {
        VoiceAssistantOptionsValidator.Validate(_options, "options");
        ValidateAudioFormat();
        ValidateOptionalCapabilities();
        if (OnlineSpeakerInstalled)
        {
            VoiceAssistantOptionsValidator.ValidateSpeakerGeometry(
                _options,
                _speaker!,
                EffectiveSpeakerPurposes,
                "options");
        }
        ValidateCapabilityEdges(CreateCapabilityEdges());

        var speechSynthesis = new SpeechSynthesisStreamAdapter(
            _speechSynthesis,
            _options.TtsVoice,
            _options.TtsSpeed);
        var audioFormat = new AssistantAudioFormat(_audio.Input, speechSynthesis);
        var runtime = new AssistantRuntimePlan(
            _audio,
            new VadTurnBoundaryAdapter(_turnBoundary),
            new OfflineRecognitionProviderAdapter(_recognition),
            _toolRuntimeFactory is null
                ? new TextOnlyResponseGeneratorFactory(_responseGeneration)
                : new ToolEnabledResponseGeneratorFactory(
                    (IToolCallingStreamLLMService)_responseGeneration,
                    _toolRuntimeFactory,
                    _options.Tools),
            speechSynthesis,
            _wakeup is null
                ? null
                : new WakeupModuleFactory(_wakeup, audioFormat, _options),
            OnlineSpeakerInstalled
                ? new SpeakerModuleFactory(_speaker!, _options.Speaker, EffectiveSpeakerPurposes)
                : null,
            _acousticDiagnostics is null
                ? null
                : new AcousticProcessingDiagnosticsAdapter(_acousticDiagnostics),
            _options.BargeIn.Enable
                ? new RenderAwarePlaybackBargeInDetectorFactory(
                    _options.BargeIn,
                    _acousticPathDelayEstimatorFactory!,
                    _detectorLogger,
                    _timeProvider)
                : null);
        runtime.Ownership = CreateOwnershipPlan();
        ModuleWiring wiring = ModuleWiring.CreateDefault(runtime);
        IReadOnlyList<AssistantLifecycleParticipant> lifecycle = CreateLifecyclePlan();
        return new AssistantComposition(
            runtime,
            wiring,
            new ModuleLifecyclePlan(lifecycle),
            CreateSnapshot());
    }

    private void ValidateAudioFormat()
    {
        IAssistantAudioInput input = _audio.Input;
        IAssistantAudioOutput output = _audio.Output;
        if (input.SampleRate == 0 || input.ChannelCount == 0 ||
            output.SampleRate == 0 || output.ChannelCount == 0)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-005",
                "Audio capture and playback must declare non-zero sample rate and channel count.");
        }

        if (!_audio.SupportsSimultaneousCapturePlayback &&
            (input.SampleRate != output.SampleRate ||
             input.ChannelCount != output.ChannelCount ||
             input.SampleFormat != output.SampleFormat))
        {
            throw new AssistantCompositionException(
                "VBX-COMP-006",
                "Split audio capture and playback formats must match before Runtime startup.");
        }
    }

    private void ValidateOptionalCapabilities()
    {
        if (_options.Wakeup.Enable && _wakeup is null)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-003",
                "Wakeup is requested but no typed IWakeupService registration was provided.");
        }

        if (_options.Speaker.Confirmation?.Enable == true)
        {
            if (_speaker is null)
            {
                throw new AssistantCompositionException(
                    "VBX-COMP-004",
                    "Speaker.Confirmation.Enable requires owner-speaker verification but no typed ISpeakerService registration was provided.");
            }
            if (!_audio.SupportsSpeakerCaptureTimeline || _audio.SpeakerCapture is null)
            {
                throw new AssistantCompositionException(
                    "VBX-COMP-013",
                    "Online speaker verification requires the identity capture tap (after the capture processor, before any independent DNR) with its own source timeline; the selected audio frontend does not provide it.");
            }
        }

        if (_options.BargeIn.Enable &&
            (!_audio.SupportsSimultaneousCapturePlayback ||
             !_audio.SupportsRenderReference ||
             !_audio.SupportsProcessedCaptureTimeline))
        {
            throw new AssistantCompositionException(
                "VBX-COMP-008",
                "Barge-in requires simultaneous capture/playback, render reference, and processed capture timeline.");
        }

        if (_options.BargeIn.Enable && _acousticPathDelayEstimatorFactory is null)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-012",
                "Barge-in requires an IAcousticPathDelayEstimatorFactory registration.");
        }

        if (EffectiveSpeakerPurposes.SingleTalk && !_audio.SupportsProcessedCaptureTimeline)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-015",
                "Speaker.Confirmation.Enable requires a locatable processed-capture source timeline so VAD chunks can be mapped to the identity query range.");
        }

        if (_wakeup is not null && !_audio.SupportsProcessedCaptureTimeline)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-009",
                "Wakeup requires the declared processed capture timeline.");
        }

        if (_requestedCapabilities.Contains(AssistantCapability.StructuredToolUse) &&
            _toolRuntimeFactory is null)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-011",
                "StructuredToolUse was requested but UseMcpTools did not install a tool runtime.");
        }

        if (_toolRuntimeFactory is not null && _responseGeneration is not IToolCallingStreamLLMService)
        {
            throw new AssistantCompositionException(
                "VBX-COMP-011",
                "UseMcpTools requires a response provider that implements " +
                "IToolCallingStreamLLMService.");
        }
    }

    /// <summary>
    /// Start order: every consumer of audio is ready before the frontend produces its first frame,
    /// so no identity or keyword evidence is lost to a start-up window; stop runs strictly in reverse
    /// (audio first). Only effectively enabled modules participate.
    /// </summary>
    private IReadOnlyList<AssistantLifecycleParticipant> CreateLifecyclePlan()
    {
        var participants = new List<AssistantLifecycleParticipant>();
        if (_toolRuntimeFactory is not null)
            participants.Add(AssistantLifecycleParticipant.ResponseGeneration);
        if (OnlineSpeakerInstalled)
            participants.Add(AssistantLifecycleParticipant.SpeakerEvidence);
        if (_wakeup is not null)
            participants.Add(AssistantLifecycleParticipant.WakeupGate);
        if (_options.BargeIn.Enable)
            participants.Add(AssistantLifecycleParticipant.BargeIn);
        participants.Add(AssistantLifecycleParticipant.AudioFrontend);
        return Array.AsReadOnly(participants.ToArray());
    }

    private AssistantCompositionSnapshot CreateSnapshot()
    {
        var capabilities = new List<AssistantCapability>
        {
            AssistantCapability.AudioCapture,
            AssistantCapability.AudioPlayback,
            AssistantCapability.CaptureTimeline,
            AssistantCapability.TurnBoundary,
            AssistantCapability.SpeechRecognition,
            AssistantCapability.ResponseGeneration,
            AssistantCapability.SpeechSynthesis,
        };
        if (_audio.SupportsSimultaneousCapturePlayback)
            capabilities.Add(AssistantCapability.SimultaneousCapturePlayback);
        if (_audio.SupportsRenderReference)
            capabilities.Add(AssistantCapability.RenderReference);
        if (_audio.SupportsProcessedCaptureTimeline)
            capabilities.Add(AssistantCapability.ProcessedCaptureTimeline);
        if (_audio.SupportsProcessedCaptureTimeline)
            capabilities.Add(AssistantCapability.CaptureProcessorStage);
        if (_audio.SupportsSpeakerCaptureTimeline)
            capabilities.Add(AssistantCapability.SpeakerCaptureTimeline);
        if (_acousticDiagnostics is not null)
            capabilities.Add(AssistantCapability.AcousticProcessingDiagnostics);
        if (_wakeup is not null)
            capabilities.Add(AssistantCapability.WakeupGate);
        if (OnlineSpeakerInstalled)
            capabilities.Add(AssistantCapability.SpeakerEvidence);
        if (_options.BargeIn.Enable)
            capabilities.Add(AssistantCapability.BargeIn);
        if (_toolRuntimeFactory is not null)
            capabilities.Add(AssistantCapability.StructuredToolUse);

        var edges = CreateCapabilityEdges();

        string[] diagnosticCapabilities = capabilities.Select(ToDiagnosticId).ToArray();
        return new AssistantCompositionSnapshot(
            _profile.ToString(),
            Array.AsReadOnly(diagnosticCapabilities),
            ComputeCapabilityGraphHash(diagnosticCapabilities))
        {
            CapabilityEdges = Array.AsReadOnly(edges
                .Select(static edge =>
                    $"{edge.ModuleId}:{edge.Kind}:{edge.Capability}:{edge.Order}")
                .ToArray()),
            Ownership = CreateOwnershipPlan(),
        };
    }

    internal static string ComputeCapabilityGraphHash(IEnumerable<string> capabilities)
    {
        string canonical = string.Join('\n', capabilities.Order(StringComparer.Ordinal));
        return ComputeHex(canonical);
    }

    private static string ComputeHex(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value)))
            .ToLowerInvariant();
    }

    internal IReadOnlyList<CapabilityEdge> CreateCapabilityEdges()
    {
        var edges = new List<CapabilityEdge>
        {
            new("AudioFrontend", AssistantCapability.AudioCapture, CapabilityEdgeKind.Provides),
            new("AudioFrontend", AssistantCapability.AudioPlayback, CapabilityEdgeKind.Provides),
            new("AudioFrontend", AssistantCapability.CaptureTimeline, CapabilityEdgeKind.Provides),
            new("TurnBoundary", AssistantCapability.AudioCapture, CapabilityEdgeKind.Requires),
            new("TurnBoundary", AssistantCapability.TurnBoundary, CapabilityEdgeKind.Provides),
            new("Recognition", AssistantCapability.TurnBoundary, CapabilityEdgeKind.Requires),
            new("Recognition", AssistantCapability.SpeechRecognition, CapabilityEdgeKind.Provides),
            new("ResponseGeneration", AssistantCapability.SpeechRecognition, CapabilityEdgeKind.Requires),
            new("ResponseGeneration", AssistantCapability.ResponseGeneration, CapabilityEdgeKind.Provides),
            new("SpeechSynthesis", AssistantCapability.ResponseGeneration, CapabilityEdgeKind.Requires),
            new("SpeechSynthesis", AssistantCapability.SpeechSynthesis, CapabilityEdgeKind.Provides),
        };

        if (_audio.SupportsSimultaneousCapturePlayback)
            edges.Add(new("AudioFrontend", AssistantCapability.SimultaneousCapturePlayback, CapabilityEdgeKind.Provides));
        if (_audio.SupportsRenderReference)
            edges.Add(new("AudioFrontend", AssistantCapability.RenderReference, CapabilityEdgeKind.Provides));
        if (_audio.SupportsProcessedCaptureTimeline)
        {
            edges.Add(new("AudioFrontend", AssistantCapability.ProcessedCaptureTimeline, CapabilityEdgeKind.Provides));
            edges.Add(new("AudioFrontend", AssistantCapability.CaptureProcessorStage, CapabilityEdgeKind.Provides));
        }
        if (_audio.SupportsSpeakerCaptureTimeline)
            edges.Add(new("AudioFrontend", AssistantCapability.SpeakerCaptureTimeline, CapabilityEdgeKind.Provides));
        if (_acousticDiagnostics is not null)
            edges.Add(new("AudioFrontend", AssistantCapability.AcousticProcessingDiagnostics, CapabilityEdgeKind.Provides));
        if (_options.BargeIn.Enable)
        {
            edges.Add(new("BargeIn", AssistantCapability.SimultaneousCapturePlayback, CapabilityEdgeKind.Requires));
            edges.Add(new("BargeIn", AssistantCapability.RenderReference, CapabilityEdgeKind.Requires));
            edges.Add(new("BargeIn", AssistantCapability.ProcessedCaptureTimeline, CapabilityEdgeKind.Requires));
            if (OnlineSpeakerInstalled)
                edges.Add(new("BargeIn", AssistantCapability.SpeakerEvidence, CapabilityEdgeKind.Requires));
            edges.Add(new("BargeIn", AssistantCapability.BargeIn, CapabilityEdgeKind.Provides));
        }
        if (_wakeup is not null)
        {
            edges.Add(new("Wakeup", AssistantCapability.AudioCapture, CapabilityEdgeKind.Requires));
            edges.Add(new("Wakeup", AssistantCapability.CaptureTimeline, CapabilityEdgeKind.Requires));
            if (OnlineSpeakerInstalled)
                edges.Add(new("Wakeup", AssistantCapability.SpeakerEvidence, CapabilityEdgeKind.Requires));
            edges.Add(new("Wakeup", AssistantCapability.WakeupGate, CapabilityEdgeKind.Provides));
        }
        if (EffectiveSpeakerPurposes.SingleTalk)
        {
            edges.Add(new("TurnBoundary", AssistantCapability.ProcessedCaptureTimeline, CapabilityEdgeKind.Requires));
        }
        if (OnlineSpeakerInstalled)
        {
            // The identity tap, never the final processed capture: speaker evidence must not wait
            // for, or be shaped by, an independent DNR.
            edges.Add(new("Speaker", AssistantCapability.SpeakerCaptureTimeline, CapabilityEdgeKind.Requires));
            edges.Add(new("Speaker", AssistantCapability.SpeakerEvidence, CapabilityEdgeKind.Provides));
        }
        if (_toolRuntimeFactory is not null)
        {
            edges.Add(new("StructuredToolUse", AssistantCapability.ResponseGeneration, CapabilityEdgeKind.Requires));
            edges.Add(new("StructuredToolUse", AssistantCapability.StructuredToolUse, CapabilityEdgeKind.Provides));
            edges.Add(new("SpeechSynthesis", AssistantCapability.StructuredToolUse, CapabilityEdgeKind.Requires));
        }
        return Array.AsReadOnly(edges.ToArray());
    }

    private static void ValidateCapabilityEdges(IReadOnlyList<CapabilityEdge> edges)
    {
        var provided = edges
            .Where(static edge => edge.Kind == CapabilityEdgeKind.Provides)
            .Select(static edge => edge.Capability)
            .ToHashSet();
        foreach (CapabilityEdge requirement in edges.Where(static edge => edge.Kind == CapabilityEdgeKind.Requires))
        {
            if (!provided.Contains(requirement.Capability))
            {
                throw new AssistantCompositionException(
                    "VBX-COMP-010",
                    $"Module '{requirement.ModuleId}' requires capability '{requirement.Capability}', but no module provides it.");
            }
        }

        AssistantCapability[] uniqueCapabilities =
        [
            AssistantCapability.TurnBoundary,
            AssistantCapability.SpeechRecognition,
            AssistantCapability.ResponseGeneration,
            AssistantCapability.SpeechSynthesis,
            AssistantCapability.WakeupGate,
            AssistantCapability.SpeakerEvidence,
            AssistantCapability.BargeIn,
            AssistantCapability.StructuredToolUse,
        ];
        foreach (AssistantCapability capability in uniqueCapabilities)
        {
            int providers = edges.Count(edge =>
                edge.Kind == CapabilityEdgeKind.Provides && edge.Capability == capability);
            if (providers > 1)
            {
                throw new AssistantCompositionException(
                    "VBX-COMP-010",
                    $"Capability '{capability}' must have a single provider; found {providers}.");
            }
        }

        var providedByModule = new Dictionary<AssistantCapability, string>();
        foreach (CapabilityEdge edge in edges.Where(static value => value.Kind == CapabilityEdgeKind.Provides))
            providedByModule[edge.Capability] = edge.ModuleId;

        var dependents = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        foreach (CapabilityEdge requirement in edges.Where(static value => value.Kind == CapabilityEdgeKind.Requires))
        {
            if (!providedByModule.TryGetValue(requirement.Capability, out string? provider))
                continue;
            if (!dependents.TryGetValue(requirement.ModuleId, out List<string>? providers))
            {
                providers = [];
                dependents[requirement.ModuleId] = providers;
            }

            if (!providers.Contains(provider, StringComparer.Ordinal))
                providers.Add(provider);
        }

        var visiting = new HashSet<string>(StringComparer.Ordinal);
        var visited = new HashSet<string>(StringComparer.Ordinal);
        foreach (string moduleId in dependents.Keys)
            DetectCapabilityCycle(moduleId, dependents, visiting, visited);
    }

    private static void DetectCapabilityCycle(
        string moduleId,
        Dictionary<string, List<string>> dependents,
        HashSet<string> visiting,
        HashSet<string> visited)
    {
        if (!visiting.Add(moduleId))
        {
            throw new AssistantCompositionException(
                "VBX-COMP-010",
                $"Capability graph contains a cyclic start dependency at module '{moduleId}'.");
        }

        if (visited.Add(moduleId) && dependents.TryGetValue(moduleId, out List<string>? providers))
        {
            foreach (string provider in providers)
                DetectCapabilityCycle(provider, dependents, visiting, visited);
        }

        visiting.Remove(moduleId);
    }

    private IReadOnlyList<ModuleOwnership> CreateOwnershipPlan()
    {
        var ownership = new List<ModuleOwnership>
        {
            new("AudioFrontend.Record", ResourceOwnership.Borrowed),
            new("AudioFrontend.Playback", ResourceOwnership.Borrowed),
            new("TurnBoundary.Provider", ResourceOwnership.Borrowed),
            new("Recognition.Provider", ResourceOwnership.Borrowed),
            new("ResponseGeneration.Provider", ResourceOwnership.Borrowed),
            new("SpeechSynthesis.Provider", ResourceOwnership.Borrowed),
        };
        // Only an Assistant-created frontend adapter is owned; a duplex frontend borrows the
        // provider and therefore contributes no owned lifecycle participant.
        if (_audio is SplitAssistantAudioFrontend)
            ownership.Add(new ModuleOwnership("AudioFrontend.Adapter", ResourceOwnership.AssistantOwned));
        if (_wakeup is not null)
        {
            ownership.Add(new ModuleOwnership("Wakeup.Provider", ResourceOwnership.Borrowed));
            ownership.Add(new ModuleOwnership("Wakeup.Module", ResourceOwnership.AssistantOwned));
        }
        if (OnlineSpeakerInstalled)
        {
            ownership.Add(new ModuleOwnership("Speaker.Provider", ResourceOwnership.Borrowed));
            ownership.Add(new ModuleOwnership("Speaker.Module", ResourceOwnership.AssistantOwned));
        }
        if (_options.BargeIn.Enable)
            ownership.Add(new ModuleOwnership("BargeIn.Module", ResourceOwnership.AssistantOwned));
        if (_toolRuntimeFactory is not null)
            ownership.Add(new ModuleOwnership("StructuredToolUse.Runtime", ResourceOwnership.AssistantOwned));
        return Array.AsReadOnly(ownership.ToArray());
    }

    private static string ToDiagnosticId(AssistantCapability capability)
        => capability switch
        {
            AssistantCapability.AudioCapture => "AudioCapture",
            AssistantCapability.AudioPlayback => "AudioPlayback",
            AssistantCapability.CaptureTimeline => "CaptureTimeline",
            AssistantCapability.SimultaneousCapturePlayback => "SimultaneousCapturePlayback",
            AssistantCapability.RenderReference => "RenderReference",
            AssistantCapability.ProcessedCaptureTimeline => "ProcessedCaptureTimeline",
            AssistantCapability.AcousticProcessingDiagnostics => "AcousticProcessingDiagnostics",
            AssistantCapability.CaptureProcessorStage => "CaptureProcessorStage",
            AssistantCapability.SpeakerCaptureTimeline => "SpeakerCaptureTimeline",
            AssistantCapability.TurnBoundary => "TurnBoundary",
            AssistantCapability.SpeechRecognition => "SpeechRecognition",
            AssistantCapability.ResponseGeneration => "ResponseGeneration",
            AssistantCapability.SpeechSynthesis => "SpeechSynthesis",
            AssistantCapability.WakeupGate => "WakeupGate",
            AssistantCapability.SpeakerEvidence => "SpeakerEvidence",
            AssistantCapability.BargeIn => "BargeIn",
            AssistantCapability.StructuredToolUse => "StructuredToolUse",
            _ => throw new ArgumentOutOfRangeException(nameof(capability), capability, null),
        };
}
