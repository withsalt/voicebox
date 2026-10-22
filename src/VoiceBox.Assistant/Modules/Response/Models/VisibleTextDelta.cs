using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.Response.Models;

internal sealed record VisibleTextDelta(
    string Text,
    bool ForceSpeechBoundary = false,
    TaskCompletionSource? SpeechBoundaryCompletion = null) : ModelTurnEvent;
