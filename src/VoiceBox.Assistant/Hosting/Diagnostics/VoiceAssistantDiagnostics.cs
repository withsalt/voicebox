using System.Collections.ObjectModel;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Hosting.Models;

namespace VoiceBox.Assistant.Hosting.Diagnostics;

/// <summary>
/// Sanitized, read-only host diagnostics. Lifecycle, Turn and module health remain separate views.
/// </summary>
public sealed class VoiceAssistantDiagnostics
{
    private readonly VoiceAssistant _assistant;
    private readonly string _profileId;
    private readonly string _capabilityGraphHash;
    private readonly IReadOnlyDictionary<string, string> _moduleStatuses;
    private string _lifecycleStatus = "Created";

    internal VoiceAssistantDiagnostics(
        VoiceAssistant assistant,
        string profileId,
        string capabilityGraphHash,
        IReadOnlyList<string> capabilities)
    {
        _assistant = assistant ?? throw new ArgumentNullException(nameof(assistant));
        _profileId = profileId ?? throw new ArgumentNullException(nameof(profileId));
        _capabilityGraphHash = capabilityGraphHash
            ?? throw new ArgumentNullException(nameof(capabilityGraphHash));

        var statuses = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["AudioFrontend"] = "Ready",
        };
        foreach (string capability in capabilities)
        {
            if (capability is "WakeupGate" or "SpeakerEvidence" or "BargeIn")
                statuses[capability] = "Ready";
        }
        _moduleStatuses = new ReadOnlyDictionary<string, string>(statuses);
    }

    public VoiceAssistantRuntimeLifecycleSnapshot RuntimeLifecycle => new(
        _profileId,
        _capabilityGraphHash,
        Volatile.Read(ref _lifecycleStatus));

    public VoiceAssistantTurnSnapshot Turn => new(_assistant.State, _assistant.WakeMode);

    public VoiceAssistantModuleHealthSnapshot Modules => new(_moduleStatuses);

    internal void SetLifecycleStatus(string status)
        => Volatile.Write(ref _lifecycleStatus, status);
}
