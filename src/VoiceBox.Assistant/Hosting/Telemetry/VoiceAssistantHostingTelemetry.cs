using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Hosting.Telemetry;

internal static class VoiceAssistantHostingTelemetry
{
    public static readonly ActivitySource ActivitySource = new("VoiceBox.Assistant.Hosting");
    public static readonly Meter Meter = new("VoiceBox.Assistant.Hosting");
    public static readonly Counter<long> Started = Meter.CreateCounter<long>(
        "voicebox.assistant.started");
    public static readonly Counter<long> Stopped = Meter.CreateCounter<long>(
        "voicebox.assistant.stopped");
    public static readonly Counter<long> Faulted = Meter.CreateCounter<long>(
        "voicebox.assistant.faulted");
    public static readonly Counter<long> TurnsCompleted = Meter.CreateCounter<long>(
        "voicebox.assistant.turns_completed");
    public static readonly Counter<long> StateTransitions = Meter.CreateCounter<long>(
        "voicebox.assistant.state_transitions");
}
