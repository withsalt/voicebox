namespace VoiceBox.Assistant.Kernel.Signals;

internal enum TurnSignalKind
{
    InputGateChanged,
    WakeupDetected,
    SpeechStarted,
    SpeechEnded,
    AuthorizationResolved,
    RecognitionSessionReady,
    RecognitionCompleted,
    RecognitionFailed,
    InterruptionRequested,
    InterruptionResolved,
    ResponseGenerated,
    PlaybackStarted,
    PlaybackCompleted,
    OperationFailed,
    CancelRequested,
    RuntimeFaultObserved,
    BackgroundResponseStarted,
}
