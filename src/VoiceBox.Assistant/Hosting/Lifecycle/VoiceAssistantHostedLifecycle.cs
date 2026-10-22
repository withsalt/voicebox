using System.Diagnostics;
using System.Diagnostics.Metrics;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Hosting.Diagnostics;
using VoiceBox.Assistant.Hosting.Telemetry;

namespace VoiceBox.Assistant.Hosting.Lifecycle;

internal sealed class VoiceAssistantHostedLifecycle : IHostedService
{
    private readonly VoiceAssistant _assistant;
    private readonly VoiceAssistantDiagnostics _diagnostics;
    private readonly ILogger<VoiceAssistantHostedLifecycle> _logger;
    private VoiceAssistantRuntimeTelemetryObserver? _runtimeTelemetry;

    public VoiceAssistantHostedLifecycle(
        VoiceAssistant assistant,
        VoiceAssistantDiagnostics diagnostics,
        ILogger<VoiceAssistantHostedLifecycle> logger)
    {
        _assistant = assistant ?? throw new ArgumentNullException(nameof(assistant));
        _diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        using Activity? activity = VoiceAssistantHostingTelemetry.ActivitySource.StartActivity(
            "voicebox.assistant.start",
            ActivityKind.Internal);
        _runtimeTelemetry = new VoiceAssistantRuntimeTelemetryObserver(_assistant);
        _diagnostics.SetLifecycleStatus("Starting");
        try
        {
            await _assistant.StartAsync(cancellationToken).ConfigureAwait(false);
            _diagnostics.SetLifecycleStatus("Running");
            VoiceAssistantHostingTelemetry.Started.Add(1);
            _logger.LogInformation(
                VoiceAssistantHostingEventIds.Started,
                "VoiceBox Assistant started with profile {ProfileId} and capability graph {CapabilityGraphHash}.",
                _diagnostics.RuntimeLifecycle.ProfileId,
                _diagnostics.RuntimeLifecycle.CapabilityGraphHash);
        }
        catch (Exception exception)
        {
            _diagnostics.SetLifecycleStatus("Faulted");
            VoiceAssistantHostingTelemetry.Faulted.Add(1);
            activity?.SetStatus(ActivityStatusCode.Error, exception.GetType().Name);
            _logger.LogError(
                VoiceAssistantHostingEventIds.StartFailed,
                exception,
                "VoiceBox Assistant failed during hosted startup.");
            DisposeRuntimeTelemetry();
            throw;
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        using Activity? activity = VoiceAssistantHostingTelemetry.ActivitySource.StartActivity(
            "voicebox.assistant.stop",
            ActivityKind.Internal);
        _diagnostics.SetLifecycleStatus("Stopping");
        try
        {
            await _assistant.StopAsync().ConfigureAwait(false);
            _diagnostics.SetLifecycleStatus("Stopped");
            VoiceAssistantHostingTelemetry.Stopped.Add(1);
            _logger.LogInformation(
                VoiceAssistantHostingEventIds.Stopped,
                "VoiceBox Assistant stopped.");
        }
        catch (Exception exception)
        {
            _diagnostics.SetLifecycleStatus("Faulted");
            VoiceAssistantHostingTelemetry.Faulted.Add(1);
            activity?.SetStatus(ActivityStatusCode.Error, exception.GetType().Name);
            _logger.LogError(
                VoiceAssistantHostingEventIds.StopFailed,
                exception,
                "VoiceBox Assistant failed during hosted shutdown.");
            throw;
        }
        finally
        {
            DisposeRuntimeTelemetry();
        }
    }

    private void DisposeRuntimeTelemetry()
    {
        VoiceAssistantRuntimeTelemetryObserver? observer = Interlocked.Exchange(ref _runtimeTelemetry, null);
        observer?.Dispose();
    }
}
