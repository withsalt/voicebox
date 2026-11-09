using VoiceBox.Assistant.Conversation.Internal.Execution;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools;

internal sealed record ToolCallRequested(string CallId, ConversationToolId ToolId, OwnedJsonValue Arguments);
