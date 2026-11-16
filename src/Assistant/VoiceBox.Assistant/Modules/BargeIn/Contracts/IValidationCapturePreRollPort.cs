namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// One-way handoff for PCM accumulated by a playback-validation owner before recognition session
/// admission. The handoff removes the buffers from the validation owner, so only RecognitionModule
/// retains them after the boundary is crossed.
/// </summary>
internal interface IValidationCapturePreRollPort
{
    bool TryTakePendingAudio(out List<float[]> audio);
}
