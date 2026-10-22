namespace VoiceBox.Assistant.Modules.TurnBoundary.Contracts;

/// <summary>
/// Locates an unchanged VAD output chunk on the retained processed-capture source timeline. The
/// identity ring and the processed reservoir share that timeline, so a located chunk yields the
/// exact interval a speaker query must cover. Both the Idle single-talk endpoint and the Speaking
/// near-end endpoint resolve their ranges through this contract; neither uses
/// <c>VADEventArgs.Data</c> without a timeline.
/// </summary>
internal interface ISpeechTimelineLocator
{
    bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample);
}
