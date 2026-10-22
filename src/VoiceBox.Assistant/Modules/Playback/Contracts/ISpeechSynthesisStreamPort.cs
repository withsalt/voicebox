using VoiceBox.Assistant.Modules.Playback.Models;

namespace VoiceBox.Assistant.Modules.Playback.Contracts;

internal interface ISpeechSynthesisStreamPort
{
    int OutputSampleRate { get; }

    int OutputChannels { get; }

    IAsyncEnumerable<SynthesizedAudioChunk> SynthesizeAsync(
        string text,
        CancellationToken cancellationToken);
}
