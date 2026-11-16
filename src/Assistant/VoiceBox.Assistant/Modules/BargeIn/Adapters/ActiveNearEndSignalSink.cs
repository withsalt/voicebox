using System.Diagnostics;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;

namespace VoiceBox.Assistant.Modules.BargeIn.Adapters;

/// <summary>
/// Bridges the render-aware near-end data plane to the control plane. Detection facts are admitted
/// as correlated signals; the confirmation itself is resolved on the control consumer so a
/// duplex callback never reduces a turn.
/// </summary>
internal sealed class ActiveNearEndSignalSink : IActiveNearEndSignalSink
{
    private readonly AssistantControlIngress _ingress;
    private readonly ControlOperationRegistry _operations;
    private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;

    public ActiveNearEndSignalSink(
        AssistantControlIngress ingress,
        ControlOperationRegistry operations,
        VoiceAssistantBenchmarkProbe? benchmarkProbe)
    {
        _ingress = ingress ?? throw new ArgumentNullException(nameof(ingress));
        _operations = operations ?? throw new ArgumentNullException(nameof(operations));
        _benchmarkProbe = benchmarkProbe;
    }

    public long NextOperationId() => _operations.NextOperationId();

    public bool TryEnqueueConfirmation(long operationId)
    {
        _benchmarkProbe?.ObserveControlTrace("near-end.admission.begin", operationId: operationId);
        _operations.ArmInterruptionAdmission();
        bool accepted = _ingress.TryEnqueueCritical(
            AssistantSignal.ActiveNearEndInterruptionRequested(operationId),
            Stopwatch.GetTimestamp(),
            AssistantSignalSource.DuplexCallback,
            out _);
        if (!accepted)
        {
            _operations.DisarmInterruptionAdmission();
            if (!_ingress.HasTerminalFault)
                Terminate("VBX-CTRL-012", "confirmation-admission-failed");
        }
        else
        {
            _benchmarkProbe?.ObserveControlTrace("near-end.admission.accepted", operationId: operationId);
        }

        return accepted;
    }

    public void EnqueueSpeechEnded(long turnId, TimeSpan duration)
    {
        if (_ingress.TryEnqueueCritical(
                AssistantSignal.ActiveNearEndSpeechEnded(turnId, duration),
                Stopwatch.GetTimestamp(),
                AssistantSignalSource.DuplexCallback,
                out _))
        {
            return;
        }

        Terminate("VBX-CTRL-013", "speech-ended-admission-failed");
    }

    public void ReportDataPlaneFault(AssistantFailure failure)
        => _ingress.TryTerminate(failure, Stopwatch.GetTimestamp(), out _);

    private void Terminate(string code, string messageKey)
        => _ingress.TryTerminate(
            new AssistantFailure(code, "capacity", "active-near-end-data-plane", messageKey),
            Stopwatch.GetTimestamp(),
            out _);
}
