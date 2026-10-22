namespace VoiceBox.Assistant.Composition.Plans;

internal enum AssistantCapability
{
    AudioCapture,
    AudioPlayback,
    CaptureTimeline,
    SimultaneousCapturePlayback,
    RenderReference,
    ProcessedCaptureTimeline,
    AcousticProcessingDiagnostics,
    CaptureProcessorStage,
    SpeakerCaptureTimeline,
    TurnBoundary,
    SpeechRecognition,
    ResponseGeneration,
    SpeechSynthesis,
    WakeupGate,
    SpeakerEvidence,
    BargeIn,
    StructuredToolUse,
}
