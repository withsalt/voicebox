using Microsoft.Extensions.Logging;

namespace VoiceBox.Assistant.Runtime.Telemetry;

/// <summary>
/// Structured EventIds for the Assistant runtime shell. Message templates never carry user text,
/// prompts, speaker embeddings or secrets.
/// </summary>
/// <remarks>
/// <c>ActivitySource</c> and <c>Meter</c> live in the <c>Hosting</c> namespace so the runtime and
/// kernel take no diagnostics dependency; observation-only instrumentation attaches at the hosting
/// boundary instead.
/// </remarks>
internal static class AssistantRuntimeEventIds
{
    public static readonly EventId Started = new(9000, nameof(Started));
    public static readonly EventId Stopped = new(9001, nameof(Stopped));
    public static readonly EventId TurnCompleted = new(9002, nameof(TurnCompleted));
    public static readonly EventId TerminalFault = new(9090, nameof(TerminalFault));
}
