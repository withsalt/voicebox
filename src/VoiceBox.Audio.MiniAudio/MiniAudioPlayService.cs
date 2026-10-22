using MiniAudioEx.Native;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.MiniAudio.Backend;
using VoiceBox.Audio.MiniAudio.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.MiniAudio;

public sealed class MiniAudioPlayService : MiniAudioServiceBase, IAudioPlayService
{
    private readonly IOptions<MiniAudioPlayOptions> _options;
    private readonly MiniAudioPlaybackQueue _queue;
    private MiniAudioDeviceSession? _session;
    private int _currentDeviceIndex = -1;
    private float _targetPlaybackGain = 1f;
    private float _currentPlaybackGain = 1f;

    public MiniAudioPlayService(IOptions<MiniAudioPlayOptions> options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        ValidateBufferOptions(_options.Value);
        if (_options.Value.MaxQueueSize is < 10 or > 10000)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "MaxQueueSize must be between 10 and 10000.");
        }

        _queue = new MiniAudioPlaybackQueue(_options.Value.MaxQueueSize);
    }

    public override AudioOptionsBase Options => _options.Value;

    public bool IsPlaying => _queue.IsPlaying;

    public int CurrentQueueSize => _queue.Count;

    public event EventHandler? OnPlaybackCompleted;

    public event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow;

    public event EventHandler<AudioErrorEventArgs>? OnError;

    public event EventHandler<AudioXrunEventArgs>? OnXrun
    {
        // MiniAudio does not expose backend xrun status flags.
        add { }
        remove { }
    }

    public int GetDefaultOutputDevice() => GetDefaultDeviceIndex(AudioDeviceType.Output);

    public async Task StartPlayingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
    {
        CheckDisposed();
        await _asyncLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await StartInternalAsync(deviceIndex, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    public async Task StopPlayingAsync(CancellationToken cancellationToken = default)
    {
        CheckDisposed();
        await _asyncLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await StopInternalAsync().ConfigureAwait(false);
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    public async Task SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken = default)
    {
        CheckDisposed();
        await _asyncLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_state != AudioServiceState.Active)
            {
                throw new InvalidOperationException("Cannot switch the output device while playback is stopped.");
            }

            if (deviceIndex == _currentDeviceIndex)
            {
                return;
            }

            await StopInternalAsync().ConfigureAwait(false);
            await StartInternalAsync(deviceIndex, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            RaiseError(exception, $"Failed to switch output device to {deviceIndex}.");
            throw;
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    public bool TryEnqueue(ReadOnlySpan<float> data, out int droppedSamples)
    {
        CheckDisposed();
        if (data.IsEmpty)
        {
            droppedSamples = 0;
            return true;
        }

        if (_state != AudioServiceState.Active)
        {
            droppedSamples = data.Length;
            return false;
        }

        bool enqueued = _queue.TryEnqueue(data, out droppedSamples);
        if (!enqueued)
        {
            MiniAudioServiceHelpers.InvokeEventHandlers(
                OnQueueOverflow,
                this,
                new AudioQueueOverflowEventArgs(droppedSamples, CurrentQueueSize, (int)_options.Value.MaxQueueSize));
        }

        return enqueued;
    }

    public void ClearQueue()
    {
        CheckDisposed();
        _asyncLock.Wait();
        try
        {
            _session?.Stop();
            _queue.Clear();
            _currentPlaybackGain = 1f;
            Volatile.Write(ref _targetPlaybackGain, 1f);
            _session?.Start();
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    public void SetPlaybackGain(float gain)
    {
        if (!float.IsFinite(gain))
        {
            throw new ArgumentOutOfRangeException(nameof(gain));
        }

        Volatile.Write(ref _targetPlaybackGain, Math.Clamp(gain, 0f, 1f));
    }

    public override void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            StopPlayingAsync().GetAwaiter().GetResult();
        }
        catch
        {
        }

        _queue.Dispose();
        base.Dispose();
    }

    private async Task StartInternalAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        if (_state == AudioServiceState.Active)
        {
            return;
        }

        if (_state is AudioServiceState.Initializing or AudioServiceState.Stopping)
        {
            throw new InvalidOperationException($"Cannot start playback while in {_state} state.");
        }

        _state = AudioServiceState.Initializing;
        try
        {
            _queue.Clear();
            MiniAudioDeviceSelection output = ResolveDevice(deviceIndex, AudioDeviceType.Output);
            _session = new MiniAudioDeviceSession(
                output,
                null,
                _options.Value.SampleRate,
                _options.Value.ChannelCount,
                _options.Value.MaxFrameSize,
                _options.Value.SuggestedLatency,
                _options.Value.EnableWasapiExclusiveMode,
                DeviceCallback);
            _session.Start();
            await Task.Delay(20, cancellationToken).ConfigureAwait(false);
            _currentDeviceIndex = output.AudioDevice.Id;
            _state = AudioServiceState.Active;
        }
        catch (Exception exception)
        {
            _session?.Dispose();
            _session = null;
            _queue.Clear();
            _currentDeviceIndex = -1;
            _state = AudioServiceState.Error;
            RaiseError(exception, "Failed to start playback.");
            throw;
        }
    }

    private Task StopInternalAsync()
    {
        if (_state is AudioServiceState.Idle or AudioServiceState.Stopping)
        {
            return Task.CompletedTask;
        }

        _state = AudioServiceState.Stopping;
        Exception? stopError = null;
        try
        {
            _session?.Stop();
        }
        catch (Exception exception)
        {
            stopError = exception;
        }
        finally
        {
            _session?.Dispose();
            _session = null;
            _queue.Clear();
            _currentDeviceIndex = -1;
        }

        if (stopError is not null)
        {
            _state = AudioServiceState.Error;
            RaiseError(stopError, "Failed to stop playback.");
            throw stopError;
        }

        _state = AudioServiceState.Idle;
        return Task.CompletedTask;
    }

    private unsafe void DeviceCallback(ma_device_ptr device, IntPtr output, IntPtr input, uint frameCount)
    {
        if (output == IntPtr.Zero || frameCount == 0)
        {
            return;
        }

        int sampleCount = checked((int)(frameCount * _options.Value.ChannelCount));
        try
        {
            Span<float> outputSamples = new((void*)output, sampleCount);
            bool completed = _queue.Fill(outputSamples);
            ApplyPlaybackGain(outputSamples);
            if (completed)
            {
                _ = Task.Run(() => MiniAudioServiceHelpers.InvokeEventHandlers(OnPlaybackCompleted, this));
            }
        }
        catch (Exception exception)
        {
            Span<float> outputSamples = new((void*)output, sampleCount);
            outputSamples.Clear();
            RaiseErrorAsync(exception, "Playback callback failed.");
        }
    }

    private void ApplyPlaybackGain(Span<float> output)
    {
        float target = Volatile.Read(ref _targetPlaybackGain);
        float current = _currentPlaybackGain;

        // Fast path: already at target
        if (MathF.Abs(target - current) < 0.0001f)
        {
            _currentPlaybackGain = target;
            if (target != 1f)
            {
                MiniAudioServiceHelpers.ScaleSamples(output, target);
            }
            return;
        }

        // Ramp path: calculate step once
        float rampSamples = _options.Value.SampleRate * _options.Value.ChannelCount * 0.015f;
        float step = rampSamples <= 0 ? (target - current) : (target - current) / rampSamples;

        _currentPlaybackGain = MiniAudioServiceHelpers.ApplyGainRamp(output, current, target, step);
    }

    private void RaiseError(Exception exception, string message)
        => MiniAudioServiceHelpers.InvokeEventHandlers(
            OnError,
            this,
            new AudioErrorEventArgs(exception, message, _state));

    private void RaiseErrorAsync(Exception exception, string message)
        => _ = Task.Run(() => RaiseError(exception, message));
}
