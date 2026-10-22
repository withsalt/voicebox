using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Playback.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.Assistant.Modules.Playback.Adapters;

internal sealed class SpeechSynthesisStreamAdapter : ISpeechSynthesisStreamPort
{
    private readonly IStreamTTSService _service;
    private readonly TtsSynthesisOptions? _options;

    public SpeechSynthesisStreamAdapter(
        IStreamTTSService service,
        string? voice,
        float? speed)
    {
        _service = service ?? throw new ArgumentNullException(nameof(service));
        _options = voice is null && speed is null
            ? null
            : new TtsSynthesisOptions { Voice = voice, Speed = speed };
    }

    public int OutputSampleRate => _service.OutputSampleRate;

    public int OutputChannels => _service.OutputChannels;

    public async IAsyncEnumerable<SynthesizedAudioChunk> SynthesizeAsync(
        string text,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (TtsAudioChunk chunk in _service
            .GetStreamAudioAsync(text, _options, cancellationToken)
            .WithCancellation(cancellationToken)
            .ConfigureAwait(false))
        {
            yield return new SynthesizedAudioChunk(
                chunk.Samples,
                chunk.SampleRate,
                chunk.Channels);
        }
    }
}
