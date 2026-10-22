namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal enum AssistantToolResultKind
{
    Accepted,
    Success,
    Denied,
    Cancelled,
    TimedOut,
    ToolError,
    ProtocolError,
    InputRequired,
    UnsupportedCapability,
    UnknownOutcome,
}
