using System.Collections.ObjectModel;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Hosting.Models;

public sealed record VoiceAssistantRuntimeLifecycleSnapshot(
    string ProfileId,
    string CapabilityGraphHash,
    string Status);
