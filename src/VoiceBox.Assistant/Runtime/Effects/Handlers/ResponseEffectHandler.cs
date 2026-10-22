using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Kernel.Effects;
using VoiceBox.Assistant.Modules.Response.Controllers;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.TurnContext;

namespace VoiceBox.Assistant.Runtime.Effects.Handlers;

internal sealed class ResponseEffectHandler : IResponseEffectHandler
{
    private readonly ResponseModule _responses;
    private readonly TurnContextStore _turnContexts;
    private readonly IResponseTurnResourcePort _turnResources;
    private readonly IEffectBackgroundScheduler _scheduler;
    private readonly ILogger _logger;
    private readonly VoiceAssistantBenchmarkProbe? _benchmarkProbe;
    private readonly bool _continuousConversationEnabled;
    private readonly IWakeupModule? _wakeup;

    public ResponseEffectHandler(
        ResponseModule responses,
        TurnContextStore turnContexts,
        IResponseTurnResourcePort turnResources,
        IEffectBackgroundScheduler scheduler,
        ILogger logger,
        bool continuousConversationEnabled,
        IWakeupModule? wakeup,
        VoiceAssistantBenchmarkProbe? benchmarkProbe = null)
    {
        _responses = responses ?? throw new ArgumentNullException(nameof(responses));
        _turnContexts = turnContexts ?? throw new ArgumentNullException(nameof(turnContexts));
        _turnResources = turnResources ?? throw new ArgumentNullException(nameof(turnResources));
        _scheduler = scheduler ?? throw new ArgumentNullException(nameof(scheduler));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _continuousConversationEnabled = continuousConversationEnabled;
        _wakeup = wakeup;
        _benchmarkProbe = benchmarkProbe;
    }

    public ValueTask GenerateResponseAsync(
        GenerateResponseEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _scheduler.Schedule(
            "response.generate",
            effect.TurnId,
            () => RunAsync(effect, cancellationToken));
        return ValueTask.CompletedTask;
    }

    public ValueTask CancelResponseAsync(
        CancelResponseEffect effect,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _benchmarkProbe?.ObserveControlTrace("effect.cancel-response", effect.TurnId, effect.OperationId);
        _responses.Cancel(effect.TurnId);
        return ValueTask.CompletedTask;
    }

    private async Task RunAsync(
        GenerateResponseEffect effect,
        CancellationToken cancellationToken)
    {
        _turnContexts.TryGetRecognitionMetadata(
            effect.TurnId,
            out string? rawText,
            out TimeSpan speechDuration);

        bool endConversationRequested = false;
        bool operationAdmitted = true;
        try
        {
            ResponseTurnResult response = await _responses.RunRecognizedTextTurnAsync(
                effect.TurnId,
                effect.Text,
                rawText ?? effect.Text,
                speechDuration,
                cancellationToken,
                effect.OperationId,
                _continuousConversationEnabled &&
                    _wakeup?.Mode == AssistantWakeMode.ContinuousConversation,
                recognitionAlreadyPublished: true,
                runtimeCancellationToken: _turnResources.RunToken,
                turnResources: _turnResources).ConfigureAwait(false);
            endConversationRequested = response.EndConversationRequested;
        }
        catch (ResponseModule.ResponseOperationAdmissionRejectedException)
        {
            operationAdmitted = false;
        }
        catch (OperationCanceledException)
        {
            _logger.LogDebug("Kernel response operation cancelled: turnId={TurnId}.", effect.TurnId);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Kernel response operation failed: turnId={TurnId}.", effect.TurnId);
        }
        finally
        {
            if (operationAdmitted)
            {
                try
                {
                    try
                    {
                        await _turnResources.PublishTurnCompletedAsync(effect.TurnId, endConversationRequested)
                            .ConfigureAwait(false);
                    }
                    catch (InvalidOperationException) when (_turnResources.IsShuttingDown)
                    {
                    }
                }
                finally
                {
                    _responses.ReleaseOperationContext(effect.TurnId);
                }
            }
        }
    }
}
