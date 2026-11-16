namespace VoiceBox.Assistant.Modules.Recognition.Contracts;

/// <summary>Prepares a recognition turn or detaches its capture session.</summary>
internal interface IRecognitionTurnLifecyclePort
{
    /// <summary>Resets per-turn capture bookkeeping before a new turn takes over.</summary>
    void PrepareTurn();

    /// <summary>Detaches the interrupted session; a promoted validation turn keeps its session.</summary>
    void ClearActiveSessionId();
}
