namespace VoiceBox.Assistant.Modules.BargeIn.Contracts;

/// <summary>
/// Validation-turn ownership operations the turn boundary needs when an endpoint lands on a
/// validation turn. Read-only fields of the same owner are exposed by
/// <see cref="IBargeInValidationTurnView"/>.
/// </summary>
internal interface IBargeInValidationOwnershipPort
{
    long ValidationTurnId { get; }

    bool HasValidation { get; }

    void ClearOwnership();
}
