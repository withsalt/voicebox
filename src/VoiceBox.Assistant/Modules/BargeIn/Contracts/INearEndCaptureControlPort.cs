namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>Capture termination used by semantic interruption owners.</summary>
internal interface INearEndCaptureControlPort
{
    bool StopCapture(long turnId, bool startCooldown);
}
