using VoiceBox.Assistant.Modules.Response.Models;

namespace VoiceBox.Assistant.Modules.Playback.Models;

internal sealed class SpeechPlaybackRequest : IDisposable
{
    public SpeechPlaybackRequest(
        long requestId,
        long? ownerTurnId,
        IAsyncEnumerable<SpeakableTextDelta> text,
        CancellationToken callerCancellation,
        CancellationToken runCancellation)
    {
        RequestId = requestId;
        OwnerTurnId = ownerTurnId;
        Text = text ?? throw new ArgumentNullException(nameof(text));
        Cancellation = CancellationTokenSource.CreateLinkedTokenSource(
            callerCancellation,
            runCancellation);
        CompletionSource = new TaskCompletionSource<string>(
            TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public long RequestId { get; }

    public long? OwnerTurnId { get; }

    public IAsyncEnumerable<SpeakableTextDelta> Text { get; }

    public CancellationTokenSource Cancellation { get; }

    public TaskCompletionSource<string> CompletionSource { get; }

    /// <summary>Set under the coordinator gate by the first owner-scoped stop; later stops are no-ops.</summary>
    public bool StopIssued { get; set; }

    /// <summary>Gain to restore after a false candidate or after this request has relinquished its device audio.</summary>
    public float? RestoreGain { get; set; }

    public void TrySetResult(string text)
        => CompletionSource.TrySetResult(text);

    public void TrySetCanceled()
        => CompletionSource.TrySetCanceled();

    public void TrySetException(Exception exception)
        => CompletionSource.TrySetException(exception);

    public void Dispose()
        => Cancellation.Dispose();
}
