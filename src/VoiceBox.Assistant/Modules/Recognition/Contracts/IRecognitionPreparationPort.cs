namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

/// <summary>
/// Prepares a provider session and tracks its deferred ready transition when an interruption is
/// accepted before ASR becomes ready. The recognition owner retains the tasks and deferred ledger.
/// </summary>
internal interface IRecognitionPreparationPort
{
    void PrepareRecognitionSession(long turnId);

    void MarkDeferred(long turnId);

    bool TakeDeferred(long turnId);
}
