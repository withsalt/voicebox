namespace VoiceBox.Assistant.Conversation.Internal.Execution;

internal interface ITurnExecutor : IAsyncDisposable
{
    IAsyncEnumerable<TurnEvent> ExecuteAsync(
        TurnExecutionContext context,
        CancellationToken cancellationToken);
}
