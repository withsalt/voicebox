namespace VoiceBox.Assistant.Modules.Conversation.Contracts;

/// <summary>
/// Records a trusted, host-generated assistant text produced by a background tool result. Raw tool
/// output never reaches this port.
/// </summary>
internal interface IConversationBackgroundTextPort
{
    bool RecordBackgroundAssistantText(long turnId, string assistantText);
}
