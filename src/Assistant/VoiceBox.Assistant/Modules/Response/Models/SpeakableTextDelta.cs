namespace VoiceBox.Assistant.Modules.Response.Models;

internal readonly record struct SpeakableTextDelta(
    string Text,
    bool ForceSpeechBoundary = false,
    TaskCompletionSource? SpeechBoundaryCompletion = null);
