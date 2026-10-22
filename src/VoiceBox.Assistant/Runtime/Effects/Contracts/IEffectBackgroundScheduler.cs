namespace VoiceBox.Assistant.Runtime.Effects.Contracts;

/// <summary>
/// Registers supervised background work for an effect. The Runtime shell owns the supervisor;
/// handlers never start unmanaged fire-and-forget tasks.
/// </summary>
internal interface IEffectBackgroundScheduler
{
    void Schedule(string taskName, long ownerTurnId, Func<Task> work);
}
