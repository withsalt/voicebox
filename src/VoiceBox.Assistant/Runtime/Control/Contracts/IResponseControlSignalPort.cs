using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Assistant.Runtime.Control.Models;

namespace VoiceBox.Assistant.Runtime.Control.Contracts;

internal interface IResponseControlSignalPort
{
    bool IsCurrentTurn(long turnId);

    /// <summary>
    /// Requests the runtime-owned input gate to open for barge-in listening. The module never
    /// mutates runtime state directly; the runtime applies its own composition and lifecycle
    /// checks before opening.
    /// </summary>
    void OpenInputForBargeIn(long turnId);

    ValueTask PublishRecognitionCompletedAsync(
        long turnId,
        string text,
        TimeSpan speechDuration,
        CancellationToken cancellationToken);

    ValueTask<ResponseControlTransition> StartResponseAsync(
        long turnId,
        long operationId,
        CancellationToken cancellationToken);

    ValueTask<ResponseControlTransition> StartSpeakableResponseAsync(
        long turnId,
        long operationId,
        CancellationToken cancellationToken);

    ValueTask PublishResponseCompletedAsync(
        long turnId,
        long operationId,
        string text,
        CancellationToken cancellationToken);

    ValueTask PublishPlaybackCompletedAsync(
        long turnId,
        long operationId,
        string text,
        TimeSpan elapsed,
        CancellationToken cancellationToken);

    ValueTask PublishResponseTerminatedAsync(
        long turnId,
        long operationId,
        ResponseOperationOutcomeKind outcome,
        string? failure,
        CancellationToken cancellationToken);
}
