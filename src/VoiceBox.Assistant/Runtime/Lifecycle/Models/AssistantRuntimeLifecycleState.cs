namespace VoiceBox.Assistant.Runtime.Lifecycle.Models;

internal enum AssistantRuntimeLifecycleState
{
    Created,
    Starting,
    Running,
    Stopping,
    Stopped,
    Faulted,
    Disposed,
}
