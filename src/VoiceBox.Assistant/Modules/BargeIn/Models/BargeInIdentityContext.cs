namespace VoiceBox.Assistant.Modules.BargeIn.Models;

/// <summary>
/// Identity facts travelling with a confirmed playback interruption. <see cref="Required"/> mirrors
/// <c>Speaker.Confirmation.Enable</c> (false means <c>NotRequired</c>); <see cref="Authorized"/> is true only when the
/// current candidate's own speaker verdict was Match. A required-but-unauthorized context can never
/// reach the Kernel as an interruption.
/// </summary>
internal readonly record struct BargeInIdentityContext(bool Required, bool Authorized, long CandidateId)
{
    public static BargeInIdentityContext NotRequired => default;

    public static BargeInIdentityContext AuthorizedSpeaker(long candidateId) => new(true, true, candidateId);

    public bool PermitsInterruption => !Required || Authorized;
}
