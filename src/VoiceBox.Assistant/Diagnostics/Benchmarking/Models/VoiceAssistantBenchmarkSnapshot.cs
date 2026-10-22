using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

namespace VoiceBox.Assistant.Diagnostics.Benchmarking.Models;

internal readonly record struct VoiceAssistantBenchmarkSnapshot(
        BargeInDecision Decision,
        float PeakNearEndProbability,
        float PeakSmoothedNearEndProbability,
        RenderAwareBargeInSnapshot Detector,
        bool PlaybackActive,
        bool RenderActive,
        ActiveNearEndDiagnosticSnapshot Coordinator,
        AssistantState AssistantState,
        bool IsSpeakingWithoutPendingCandidate);
