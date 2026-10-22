using System.Buffers;
using System.Diagnostics;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Adapters;

/// <summary>
/// Adapts an independent record/play pair. It establishes a monotonic capture sequence for the
/// processed timeline but never claims render reference or simultaneous capture/playback, because
/// it cannot prove a cross-device clock relationship. Both providers remain borrowed.
/// </summary>
internal sealed class SplitAssistantAudioFrontend :
    IAssistantAudioFrontend,
    IProcessedCaptureSource,
    IDisposable
{
    private readonly IAudioRecordService _record;
    private readonly IAudioPlayService _play;
    private readonly CaptureInputPort _input;
    private readonly PlaybackOutputPort _output;
    private long _captureSequence;
    private long _captureSample;
    private int _disposed;

    public SplitAssistantAudioFrontend(IAudioRecordService record, IAudioPlayService play)
    {
        _record = record ?? throw new ArgumentNullException(nameof(record));
        _play = play ?? throw new ArgumentNullException(nameof(play));
        _input = new CaptureInputPort(_record);
        _output = new PlaybackOutputPort(_play);
        _record.OnDataAvailable += OnRecordDataAvailable;
    }

    public IAssistantAudioInput Input => _input;

    public IAssistantAudioOutput Output => _output;

    public IProcessedCaptureSource? ProcessedCapture => this;

    public IRenderReferenceSource? RenderReference => null;

    public IDuplexAudioFrameSource? DuplexFrames => null;

    /// <summary>
    /// A record/play pair exposes no post-processor identity tap with a proven source timeline, so
    /// it must never be used for realtime speaker purposes; composition fails instead of falling
    /// back to the final processed audio.
    /// </summary>
    public SpeakerCaptureDescriptor? SpeakerCapture => null;

    public bool SupportsSimultaneousCapturePlayback => false;

    public bool SupportsRenderReference => false;

    public bool SupportsProcessedCaptureTimeline => false;

    public bool SupportsSpeakerCaptureTimeline => false;

    public event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;

    private sealed class CaptureInputPort(IAudioRecordService record) : IAssistantAudioInput
    {
        public uint SampleRate => record.Options.SampleRate;

        public uint ChannelCount => record.Options.ChannelCount;

        public SampleFormat SampleFormat => record.Options.SampleFormat;

        public bool IsCapturing => record.IsRecording;
    }

    private sealed class PlaybackOutputPort(IAudioPlayService play) : IAssistantAudioOutput
    {
        public uint SampleRate => play.Options.SampleRate;

        public uint ChannelCount => play.Options.ChannelCount;

        public SampleFormat SampleFormat => play.Options.SampleFormat;

        public bool IsPlaying => play.IsPlaying;

        public int CurrentQueueSize => play.CurrentQueueSize;

        public AudioServiceState State => play.State;

        public bool TryEnqueue(ReadOnlySpan<float> samples, out int droppedSamples)
            => play.TryEnqueue(samples, out droppedSamples);

        public void ClearQueue() => play.ClearQueue();

        public void SetGain(float gain) => play.SetPlaybackGain(gain);
    }

    /// <summary>
    /// Playback starts before capture so the first captured frame can never precede an active
    /// output; a capture failure rolls the started playback back.
    /// </summary>
    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        await _play.StartPlayingAsync(-1, cancellationToken).ConfigureAwait(false);
        try
        {
            await _record.StartRecordingAsync(-1, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            try { await _play.StopPlayingAsync(CancellationToken.None).ConfigureAwait(false); } catch { }
            throw;
        }
    }

    /// <summary>Capture stops first, then playback follows the existing drain/stop policy.</summary>
    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        Exception? captureFailure = null;
        try { await _record.StopRecordingAsync(cancellationToken).ConfigureAwait(false); }
        catch (Exception ex) { captureFailure = ex; }

        try { await _play.StopPlayingAsync(cancellationToken).ConfigureAwait(false); }
        catch when (captureFailure is not null) { }

        if (captureFailure is not null)
            throw captureFailure;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _record.OnDataAvailable -= OnRecordDataAvailable;
    }

    private void OnRecordDataAvailable(ReadOnlySpan<float> audio)
    {
        ProcessedCaptureFrameHandler? handler = OnProcessedCaptureFrame;
        if (handler is null || audio.IsEmpty)
            return;

        long sequence = Interlocked.Increment(ref _captureSequence);
        long endSample = Interlocked.Add(ref _captureSample, audio.Length);
        long startSample = checked(endSample - audio.Length);
        var range = new CaptureRange(
            StreamEpoch: 1,
            StartSequenceNumber: sequence,
            EndSequenceNumber: sequence,
            StartSample: startSample,
            EndSampleExclusive: endSample,
            TimestampTicks: Stopwatch.GetTimestamp());
        // ProcessedCaptureFrame lends its PCM only for the synchronous event callback.
        // Each invocation takes its own lease, including concurrent or reentrant callbacks.
        float[] buffer = ArrayPool<float>.Shared.Rent(audio.Length);
        try
        {
            audio.CopyTo(buffer);
            var frame = new ProcessedCaptureFrame(
                range,
                _record.Options.SampleRate,
                _record.Options.ChannelCount,
                buffer.AsMemory(0, audio.Length),
                DelaySamples: 0,
                HasDiscontinuity: false);
            handler.Invoke(in frame);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }
}
