namespace VoiceBox.Assistant.Runtime.Control.Models;

internal enum AssistantSignalSource
{
    HostLifecycle,
    CaptureCallback,
    DuplexCallback,
    Wakeup,
    Recognition,
    Response,
    Playback,
    Validation,
    TerminalFault,
}
