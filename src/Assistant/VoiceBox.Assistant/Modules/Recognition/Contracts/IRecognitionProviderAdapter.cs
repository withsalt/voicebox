namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

internal interface IRecognitionProviderAdapter
{
    Task<string> StartSessionAsync(CancellationToken cancellationToken);

    void FeedAudio(string sessionId, ReadOnlySpan<float> audio);

    Task<string?> FinishSessionAsync(string sessionId, CancellationToken cancellationToken);

    /// <summary>Releases an abandoned session when the provider supports explicit cancellation.</summary>
    Task<bool> TryCancelSessionAsync(string sessionId, CancellationToken cancellationToken)
        => Task.FromResult(false);
}
