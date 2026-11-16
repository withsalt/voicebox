namespace VoiceBox.Assistant.Modules.Speaker.Models;

internal enum SpeakerWindowEvidenceKind
{
    /// <summary>The window produced an embedding and bank scores.</summary>
    Scored,

    /// <summary>The window failed the quality gate (silence, clipping, DC) and was not inferred.</summary>
    Ineligible,

    /// <summary>The provider or timeline failed for this window; it cannot prove or disprove identity.</summary>
    Failed,
}
