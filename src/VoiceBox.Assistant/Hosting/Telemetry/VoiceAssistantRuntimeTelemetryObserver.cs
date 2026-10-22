using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Hosting.Telemetry;

/// <summary>
/// Observation-only runtime instrumentation. It subscribes to the public state and reply events
/// and records counters and activities; it never mutates state, swallows an exception, or changes
/// event order, so the business trace is identical with instrumentation enabled or disabled.
/// No user text, prompt, embedding or secret is recorded.
/// </summary>
internal sealed class VoiceAssistantRuntimeTelemetryObserver : IDisposable
{
    private readonly VoiceAssistant _assistant;
    private int _disposed;

    public VoiceAssistantRuntimeTelemetryObserver(VoiceAssistant assistant)
    {
        _assistant = assistant ?? throw new ArgumentNullException(nameof(assistant));
        _assistant.StateChanged += OnStateChanged;
        _assistant.AssistantReplied += OnAssistantReplied;
    }

    private static void OnStateChanged(object? sender, AssistantStateChangedEventArgs e)
        => VoiceAssistantHostingTelemetry.StateTransitions.Add(
            1,
            new KeyValuePair<string, object?>("from", e.From.ToString()),
            new KeyValuePair<string, object?>("to", e.To.ToString()));

    private static void OnAssistantReplied(object? sender, AssistantReplyEventArgs e)
        => VoiceAssistantHostingTelemetry.TurnsCompleted.Add(1);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _assistant.StateChanged -= OnStateChanged;
        _assistant.AssistantReplied -= OnAssistantReplied;
    }
}
