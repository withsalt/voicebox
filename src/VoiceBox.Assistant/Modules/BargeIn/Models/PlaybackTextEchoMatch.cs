namespace VoiceBox.Assistant.Modules.BargeIn.Models;

internal readonly record struct PlaybackTextEchoMatch(
        bool IsMatch,
        string Reason);
