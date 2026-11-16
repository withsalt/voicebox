namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

/// <summary>
/// Runtime-owned response turn resources. The response effect handler borrows them for exactly
/// one operation and never decides semantic state transitions.
/// </summary>
internal interface IResponseTurnResourcePort
{
    CancellationToken RunToken { get; }

    bool IsShuttingDown { get; }

    bool TryAttachResponseContext(long turnId, CancellationTokenSource responseCts);

    void ReleaseResponseContext(long turnId, CancellationTokenSource responseCts);

    ValueTask PublishTurnCompletedAsync(long turnId, bool endConversationRequested);
}
