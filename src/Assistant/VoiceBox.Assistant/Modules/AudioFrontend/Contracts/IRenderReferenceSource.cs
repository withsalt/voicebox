namespace VoiceBox.Assistant.Modules.AudioFrontend.Contracts;

/// <summary>
/// Optional synchronized render timeline exposed by a capable audio frontend. It is the same duplex
/// frame subscription as <see cref="IDuplexAudioFrameSource"/>; the marker only records that the
/// frames carry a render reference the double-talk detector can use.
/// </summary>
internal interface IRenderReferenceSource : IDuplexAudioFrameSource
{
}
