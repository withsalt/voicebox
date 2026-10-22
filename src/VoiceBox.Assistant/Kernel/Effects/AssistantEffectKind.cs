namespace VoiceBox.Assistant.Kernel.Effects;

internal enum AssistantEffectKind
{
    SetInputGate,
    PrepareRecognition,
    FinishRecognition,
    GenerateResponse,
    CancelResponse,
    ClearPlaybackQueue,
    StopPlayback,
    CommitHistory,
    RollbackHistory,
    PublishStatefulEvent,
    CancelOperation,
    CompleteTurn,
}
