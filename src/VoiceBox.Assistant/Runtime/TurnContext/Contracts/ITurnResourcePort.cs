namespace VoiceBox.Assistant.Runtime.TurnContext.Contracts;

internal interface ITurnResourcePort
{
    TurnResourceSlot GetOrCreate(long turnId);
    bool TryGet(long turnId, out TurnResourceSlot slot);
    bool TryRemoveResponseCts(CancellationTokenSource responseCts);
}
