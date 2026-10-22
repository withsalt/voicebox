using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.Response.Models;

internal sealed record ThinkingDelta(string Text) : ModelTurnEvent;
