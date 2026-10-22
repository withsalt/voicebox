namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct RuntimeCommand(RuntimeCommandKind Kind, long OperationId = 0)
{
    public bool IsBarrier => Kind is RuntimeCommandKind.Stop or RuntimeCommandKind.Dispose;
}
