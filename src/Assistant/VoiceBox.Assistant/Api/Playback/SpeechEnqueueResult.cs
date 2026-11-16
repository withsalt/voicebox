namespace VoiceBox.Assistant.Api.Playback;

/// <summary>
/// Identifies a text-to-speech request admitted to the assistant playback queue.
/// </summary>
public sealed class SpeechEnqueueResult
{
    internal SpeechEnqueueResult(long requestId, Task completion)
    {
        RequestId = requestId;
        Completion = completion ?? throw new ArgumentNullException(nameof(completion));
    }

    /// <summary>Monotonically increasing identifier for this assistant instance.</summary>
    public long RequestId { get; }

    /// <summary>
    /// Completes when this request has either finished playback or been cancelled/failed.
    /// The task does not carry the synthesized text; it is intended for lifecycle observation.
    /// </summary>
    public Task Completion { get; }
}
