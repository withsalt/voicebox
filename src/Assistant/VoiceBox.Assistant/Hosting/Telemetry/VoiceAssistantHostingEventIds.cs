using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;

namespace VoiceBox.Assistant.Hosting.Telemetry;

internal static class VoiceAssistantHostingEventIds
{
    public static readonly EventId Started = new(9100, nameof(Started));
    public static readonly EventId Stopped = new(9101, nameof(Stopped));
    public static readonly EventId StartFailed = new(9190, nameof(StartFailed));
    public static readonly EventId StopFailed = new(9191, nameof(StopFailed));
}
