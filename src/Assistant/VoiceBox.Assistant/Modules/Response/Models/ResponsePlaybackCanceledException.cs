namespace VoiceBox.Assistant.Modules.Response.Models;

/// <summary>Playback was stopped by its scheduler, including before response cancellation arrives.</summary>
internal sealed class ResponsePlaybackCanceledException(OperationCanceledException innerException)
    : OperationCanceledException("Speech playback was canceled.", innerException, innerException.CancellationToken);
