namespace VoiceBox.Assistant.Runtime.TurnContext;

internal interface ITurnResource
{
    TurnResources GetOrCreate(long turnId);

    bool TryGet(long turnId, out TurnResources slot);

    bool TryRemoveResponseCts(CancellationTokenSource responseCts);
}
