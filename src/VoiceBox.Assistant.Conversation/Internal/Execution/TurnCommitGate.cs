namespace VoiceBox.Assistant.Conversation.Internal.Execution;

/// <summary>
/// Releases background tool work only after the foreground turn has been committed. Aborting the
/// gate on rollback makes a registered background task exit without running user code.
/// </summary>
internal sealed class TurnCommitGate
{
    private readonly TaskCompletionSource<bool> _completion =
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    public Task<bool> WaitAsync() => _completion.Task;

    public void Commit() => _completion.TrySetResult(true);

    public void Abort() => _completion.TrySetResult(false);
}
