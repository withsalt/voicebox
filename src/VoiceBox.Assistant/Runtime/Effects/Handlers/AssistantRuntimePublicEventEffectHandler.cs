using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Runtime.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Assistant.Runtime.Effects.Handlers;

/// <summary>
/// Runtime-owned stateful public-event effect executor. Modules never see this type; the closed
/// effect runner reaches it through the typed public-event handler family.
/// </summary>
internal sealed class AssistantRuntimePublicEventEffectHandler : IPublicEventEffectHandler
{
    private readonly OrderedPublicationBoundary _publications;
    private readonly ILogger _logger;
    private readonly IRuntimeFaultPublicationGate _faultPublication;
    private readonly IAssistantStateProjectionSink _stateProjection;
    private readonly BargeInDecisionJournal? _bargeInJournal;
    private long _lastPublicationSequence;

    public AssistantRuntimePublicEventEffectHandler(
        OrderedPublicationBoundary publications,
        ILogger logger,
        IRuntimeFaultPublicationGate faultPublication,
        IAssistantStateProjectionSink stateProjection,
        BargeInDecisionJournal? bargeInJournal = null)
    {
        _publications = publications ?? throw new ArgumentNullException(nameof(publications));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _faultPublication = faultPublication ?? throw new ArgumentNullException(nameof(faultPublication));
        _stateProjection = stateProjection ?? throw new ArgumentNullException(nameof(stateProjection));
        _bargeInJournal = bargeInJournal;
    }

    public long LastPublicationSequence => Volatile.Read(ref _lastPublicationSequence);

    public ValueTask PublishWakeModeChangedAsync(
        PublishWakeModeChangedEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Publish(_publications.RaiseWakeModeChanged(effect.From, effect.To));
        return ValueTask.CompletedTask;
    }

    public ValueTask PublishWakeupDetectedAsync(
        PublishWakeupDetectedEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        KernelWakeupObservation value = effect.Observation;
        Publish(_publications.RaiseWakeupDetected(new AssistantWakeupDetectedEventArgs
        {
            Wakeup = new WakeupEventArgs(
                value.Keyword,
                value.Score,
                value.Threshold,
                value.DurationMs,
                value.StartTimeMs),
        }));
        return ValueTask.CompletedTask;
    }

    public ValueTask PublishStateChangedAsync(
        PublishStateChangedEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _stateProjection.Publish(effect.To);
        // A failed publication worker already emitted its terminal fault watermark. Keep
        // lifecycle state updates working so shutdown retires the run, then Drain propagates
        // the original worker fault; no later event can be delivered on that context.
        if (_publications.IsFaulted)
            return ValueTask.CompletedTask;
        Publish(_publications.RaiseStateChanged(effect.From, effect.To));
        return ValueTask.CompletedTask;
    }

    public ValueTask PublishBargeInValidationAsync(
        PublishBargeInValidationEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        // Heard carries the immutable acoustic observation in the effect because it is published
        // by the same decision that admits the journal entry. Later Accepted/Rejected effects use
        // the module-owned journal committed after that first decision.
        BargeInDecisionDiagnostics diagnostics = effect.Diagnostics;
        BargeInTriggerSource source = effect.Source;
        if (!effect.HasAcousticContext && _bargeInJournal is not null)
        {
            _bargeInJournal.TryGetDiagnostics(effect.TurnId, out diagnostics);
            source = _bargeInJournal.GetTriggerSource(effect.TurnId);
        }

        Publish(_publications.RaiseBargeInValidation(new BargeInValidationEventArgs
        {
            Kind = effect.ValidationKind,
            Text = effect.Text,
            Reason = effect.Reason,
            Source = source,
            AudioFrameCount = diagnostics.FrameCount,
            AudioAcceptedFrameCount = diagnostics.AcceptedFrameCount,
            AudioEchoLikeFrameCount = diagnostics.EchoLikeFrameCount,
            AudioAcceptedDurationMs = diagnostics.AcceptedDurationMs,
            AudioScoreDurationMs = diagnostics.ScoreDurationMs,
            AudioPeakScoreDurationMs = diagnostics.PeakScoreDurationMs,
        }));
        return ValueTask.CompletedTask;
    }

    public ValueTask PublishUserSpokeAsync(
        PublishUserSpokeEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Publish(_publications.RaiseUserSpoke(new UserSpokeEventArgs
        {
            Text = effect.Text,
            SpeechDuration = effect.SpeechDuration,
        }));
        return ValueTask.CompletedTask;
    }

    public ValueTask PublishAssistantResponseGeneratedAsync(
        PublishAssistantResponseGeneratedEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Publish(_publications.RaiseAssistantResponseGenerated(
            new AssistantResponseGeneratedEventArgs { Text = effect.Text }));
        return ValueTask.CompletedTask;
    }

    public ValueTask PublishAssistantRepliedAsync(
        PublishAssistantRepliedEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Publish(_publications.RaiseAssistantReplied(new AssistantReplyEventArgs
        {
            Text = effect.Text,
            Elapsed = effect.Elapsed,
        }));
        return ValueTask.CompletedTask;
    }

    public ValueTask PublishFaultedAsync(
        PublishFaultedEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_publications.IsFaulted)
            return ValueTask.CompletedTask;
        if (_faultPublication.TryClaim(out Exception? fatalException))
        {
            Exception exception = fatalException ?? new InvalidOperationException(
                $"{effect.Failure.Code}:{effect.Failure.Component}:{effect.Failure.MessageKey}");
            Publish(_publications.RaiseFaulted(new AssistantFaultedEventArgs
            {
                Component = effect.Failure.Component,
                Exception = exception,
            }));
        }
        return ValueTask.CompletedTask;
    }

    private void Publish(long sequence)
        => Volatile.Write(ref _lastPublicationSequence, sequence);
}
