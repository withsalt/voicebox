namespace VoiceBox.Assistant.Modules.ModelInteraction.Models;

internal enum AssistantToolRisk
{
    ReadOnly,
    Sensitive,
    SideEffecting,
    Destructive,
    Privileged,
}
