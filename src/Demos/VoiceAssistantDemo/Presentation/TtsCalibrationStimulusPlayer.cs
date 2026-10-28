using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Audio.Common.Resampling.Service;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceAssistantDemo;

/// <summary>
/// Calibration speech for controlled double-talk enrollment: a fixed sentence is synthesized once
/// with the product TTS voice, converted to the duplex device format and looped through the same
/// playback queue the assistant uses, at normal gain, until the enrollment coordinator stops it.
/// </summary>
internal sealed class TtsCalibrationStimulusPlayer : ISpeakerEnrollmentStimulusPlayer, IDisposable
{
    private const string CalibrationText =
        "现在我来念一段校准语音，请你在我说话的同时自然地朗读屏幕上的句子，不用等我停下来。" +
        "天气预报说明天多云转晴，气温十八到二十六度，适合出门散步。";

    private readonly IAudioDuplexService _audio;
    private readonly ITTSService _tts;
    private readonly string? _voice;
    private readonly float? _speed;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private float[]? _pcm;
    private CancellationTokenSource? _loop;
    private Task? _pump;

    public TtsCalibrationStimulusPlayer(IAudioDuplexService audio, ITTSService tts, string? voice, float? speed)
    {
        _audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _tts = tts ?? throw new ArgumentNullException(nameof(tts));
        _voice = voice;
        _speed = speed;
    }

    /// <summary>
    /// Synthesizes and converts the calibration speech before the user starts a double-talk
    /// segment, so model initialization does not delay playback after the start key is pressed.
    /// </summary>
    public async Task PrepareAsync(CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            _ = await EnsurePreparedAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await StopCoreAsync().ConfigureAwait(false);
            float[] pcm = await EnsurePreparedAsync(cancellationToken).ConfigureAwait(false);
            var loop = new CancellationTokenSource();
            _loop = loop;
            _pump = Task.Run(() => PumpAsync(pcm, loop.Token), CancellationToken.None);
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await StopCoreAsync().ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task StopCoreAsync()
    {
        CancellationTokenSource? loop = _loop;
        Task? pump = _pump;
        _loop = null;
        _pump = null;
        if (loop is null)
            return;

        loop.Cancel();
        if (pump is not null)
        {
            try { await pump.ConfigureAwait(false); }
            catch (OperationCanceledException) { }
        }
        loop.Dispose();
        _audio.ClearQueue();
    }

    private async Task<float[]> EnsurePreparedAsync(CancellationToken cancellationToken)
    {
        if (_pcm is not null)
            return _pcm;

        _pcm = await SynthesizeAsync(cancellationToken).ConfigureAwait(false);
        return _pcm;
    }

    private async Task<float[]> SynthesizeAsync(CancellationToken cancellationToken)
    {
        TtsResult result = await _tts.GetAudioAsync(
            CalibrationText,
            new TtsSynthesisOptions { Voice = _voice, Speed = _speed },
            cancellationToken).ConfigureAwait(false);
        int targetRate = (int)_audio.Options.SampleRate;
        int targetChannels = (int)_audio.Options.ChannelCount;
        if (result.SampleRate == targetRate && _tts.OutputChannels == targetChannels)
            return result.Samples;

        using var resampler = new AudioResampleService(new AudioResampleOptions
        {
            TargetSampleRate = targetRate,
            TargetChannels = targetChannels,
        });
        float[] converted = resampler.Convert(result.Samples, result.SampleRate, _tts.OutputChannels);
        float[] tail = resampler.Flush();
        if (tail.Length == 0)
            return converted;
        var joined = new float[converted.Length + tail.Length];
        converted.CopyTo(joined, 0);
        tail.CopyTo(joined, converted.Length);
        return joined;
    }

    /// <summary>Feeds the device queue in small blocks so a stop takes effect within one block.</summary>
    private async Task PumpAsync(float[] pcm, CancellationToken cancellationToken)
    {
        int blockSamples = Math.Max(1, (int)(_audio.Options.SampleRate * _audio.Options.ChannelCount / 10));
        int position = 0;
        while (!cancellationToken.IsCancellationRequested)
        {
            int length = Math.Min(blockSamples, pcm.Length - position);
            if (!_audio.TryEnqueue(pcm.AsSpan(position, length), out _))
            {
                await Task.Delay(20, cancellationToken).ConfigureAwait(false);
                continue;
            }

            position += length;
            if (position >= pcm.Length)
                position = 0;
            // Keep roughly half a second queued ahead of the device.
            if (_audio.CurrentQueueSize > blockSamples * 5)
                await Task.Delay(50, cancellationToken).ConfigureAwait(false);
        }
    }

    public void Dispose()
    {
        try { StopCoreAsync().GetAwaiter().GetResult(); }
        catch { }
        _gate.Dispose();
    }
}
