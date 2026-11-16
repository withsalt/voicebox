using VoiceBox.Assistant.Runtime.Models;

namespace VoiceBox.Assistant.Runtime.Contracts;

internal interface IPlaybackActivityView
{
    PlaybackActivitySnapshot GetActivitySnapshot();
}
