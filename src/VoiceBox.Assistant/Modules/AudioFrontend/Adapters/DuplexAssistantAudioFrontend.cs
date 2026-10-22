using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Adapters;

/// <summary>Adapts one duplex provider to the typed capture/playback ports.</summary>
internal sealed class DuplexAssistantAudioFrontend :
    IAssistantAudioFrontend,
    IRenderReferenceSource,
    IProcessedCaptureSource
{
    private readonly IAudioDuplexService _audio;
    private readonly DuplexInputPort _input;
    private readonly DuplexOutputPort _output;
    private readonly SpeakerCaptureDescriptor _speakerCapture;

    public DuplexAssistantAudioFrontend(
        IAudioDuplexService audio,
        IAcousticProcessingMetricsSource? acousticDiagnostics = null)
    {
        _audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _input = new DuplexInputPort(_audio);
        _output = new DuplexOutputPort(_audio);
        _speakerCapture = SpeakerCaptureDescriptor.FromDuplexService(_audio, acousticDiagnostics);
    }

    public IAssistantAudioInput Input => _input;

    public IAssistantAudioOutput Output => _output;

    public IProcessedCaptureSource? ProcessedCapture => this;

    public IRenderReferenceSource? RenderReference => this;

    public IDuplexAudioFrameSource? DuplexFrames => this;

    public SpeakerCaptureDescriptor? SpeakerCapture => _speakerCapture;

    public bool SupportsSimultaneousCapturePlayback => true;

    public bool SupportsRenderReference => true;

    public bool SupportsProcessedCaptureTimeline => true;

    public bool SupportsSpeakerCaptureTimeline => true;

    private sealed class DuplexInputPort(IAudioDuplexService audio) : IAssistantAudioInput
    {
        public uint SampleRate => audio.Options.SampleRate;

        public uint ChannelCount => audio.Options.ChannelCount;

        public SampleFormat SampleFormat => audio.Options.SampleFormat;

        public bool IsCapturing => audio.IsRecording;
    }

    private sealed class DuplexOutputPort(IAudioDuplexService audio) : IAssistantAudioOutput
    {
        public uint SampleRate => audio.Options.SampleRate;

        public uint ChannelCount => audio.Options.ChannelCount;

        public SampleFormat SampleFormat => audio.Options.SampleFormat;

        public bool IsPlaying => audio.IsPlaying;

        public int CurrentQueueSize => audio.CurrentQueueSize;

        public AudioServiceState State => ((IAudioPlayService)audio).State;

        public bool TryEnqueue(ReadOnlySpan<float> samples, out int droppedSamples)
            => audio.TryEnqueue(samples, out droppedSamples);

        public void ClearQueue() => audio.ClearQueue();

        public void SetGain(float gain) => audio.SetPlaybackGain(gain);
    }

    public Task StartAsync(CancellationToken cancellationToken = default)
        => _audio.StartAsync(cancellationToken: cancellationToken);

    public Task StopAsync(CancellationToken cancellationToken = default)
        => _audio.StopAsync(cancellationToken);

    public event DuplexAudioFrameHandler? OnDuplexAudioFrame
    {
        add => _audio.OnDuplexAudioFrame += value;
        remove => _audio.OnDuplexAudioFrame -= value;
    }

    public event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame
    {
        add => _audio.OnProcessedCaptureFrame += value;
        remove => _audio.OnProcessedCaptureFrame -= value;
    }
}
