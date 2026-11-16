namespace VoiceBox.Assistant.Kernel.Models;

internal enum ResponseLifecycle
{
    None,
    Generating,
    Generated,
    Playing,
    Completed,
    Failed,
    Cancelled,
}
