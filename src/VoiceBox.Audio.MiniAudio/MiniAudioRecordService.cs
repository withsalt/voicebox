using System.Buffers;
using System.Threading.Channels;
using MiniAudioEx.Native;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.MiniAudio.Backend;
using VoiceBox.Audio.MiniAudio.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.MiniAudio;

public sealed class MiniAudioRecordService : MiniAudioServiceBase, IAudioRecordService
{
    private readonly IOptions<MiniAudioRecordOptions> _options;
    private MiniAudioDeviceSession? _session;
    private Channel<CaptureBuffer>? _captureChannel;
    private Task? _processingTask;
    private int _currentDeviceIndex = -1;
    private volatile bool _isRecording;

    public MiniAudioRecordService(IOptions<MiniAudioRecordOptions> options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        ValidateBufferOptions(_options.Value);
        if (_options.Value.BufferCapacity is < 1 or > 32767u)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "BufferCapacity must be between 1 and 32767.");
        }
    }

    public override AudioOptionsBase Options => _options.Value;

    public bool IsRecording => _isRecording;

    public event Action<ReadOnlySpan<float>>? OnDataAvailable;

    public event EventHandler<AudioErrorEventArgs>? OnError;

    public event EventHandler<AudioXrunEventArgs>? OnXrun
    {
        // MiniAudio does not expose backend xrun status flags.
        add { }
        remove { }
    }

    public int GetDefaultInputDevice() => GetDefaultDeviceIndex(AudioDeviceType.Input);

    public async Task StartRecordingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
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

    public async Task StopRecordingAsync(CancellationToken cancellationToken = default)
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
                throw new InvalidOperationException("Cannot switch the input device while recording is stopped.");
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
            RaiseError(exception, $"Failed to switch input device to {deviceIndex}.");
            throw;
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    public override void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            StopRecordingAsync().GetAwaiter().GetResult();
        }
        catch
        {
        }

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
            throw new InvalidOperationException($"Cannot start recording while in {_state} state.");
        }

        _state = AudioServiceState.Initializing;
        try
        {
            MiniAudioDeviceSelection input = ResolveDevice(deviceIndex, AudioDeviceType.Input);
            Channel<CaptureBuffer> channel = CreateCaptureChannel(_options.Value.BufferCapacity);
            _captureChannel = channel;
            _processingTask = Task.Run(() => ProcessQueueAsync(channel));
            _session = new MiniAudioDeviceSession(
                null,
                input,
                _options.Value.SampleRate,
                _options.Value.ChannelCount,
                _options.Value.MaxFrameSize,
                _options.Value.SuggestedLatency,
                _options.Value.EnableWasapiExclusiveMode,
                DeviceCallback);
            _currentDeviceIndex = input.AudioDevice.Id;
            _isRecording = true;
            _session.Start();
            await Task.Delay(20, cancellationToken).ConfigureAwait(false);
            _state = AudioServiceState.Active;
        }
        catch (Exception exception)
        {
            _isRecording = false;
            _session?.Dispose();
            _session = null;
            _captureChannel?.Writer.TryComplete();
            await DrainPipelineAsync().ConfigureAwait(false);
            _captureChannel = null;
            _currentDeviceIndex = -1;
            _state = AudioServiceState.Error;
            RaiseError(exception, "Failed to start recording.");
            throw;
        }
    }

    private async Task StopInternalAsync()
    {
        if (_state is AudioServiceState.Idle or AudioServiceState.Stopping)
        {
            return;
        }

        _state = AudioServiceState.Stopping;
        _isRecording = false;
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
        }

        _captureChannel?.Writer.TryComplete();
        await DrainPipelineAsync().ConfigureAwait(false);
        _captureChannel = null;
        _currentDeviceIndex = -1;
        if (stopError is not null)
        {
            _state = AudioServiceState.Error;
            RaiseError(stopError, "Failed to stop recording.");
            throw stopError;
        }

        _state = AudioServiceState.Idle;
    }

    private unsafe void DeviceCallback(ma_device_ptr device, IntPtr output, IntPtr input, uint frameCount)
    {
        if (!_isRecording || input == IntPtr.Zero || frameCount == 0)
        {
            return;
        }

        int sampleCount = checked((int)(frameCount * _options.Value.ChannelCount));

        // Round up to power of 2 for better ArrayPool bucketing
        int rentSize = sampleCount <= 16 ? sampleCount : RoundUpToPowerOf2(sampleCount);
        float[] buffer = ArrayPool<float>.Shared.Rent(rentSize);

        try
        {
            new ReadOnlySpan<float>((void*)input, sampleCount).CopyTo(buffer);
            var item = new CaptureBuffer(buffer, sampleCount);
            buffer = [];
            Channel<CaptureBuffer>? channel = _captureChannel;
            if (channel is null || !channel.Writer.TryWrite(item))
            {
                item.Release();
            }
        }
        catch (Exception exception)
        {
            if (buffer.Length > 0)
            {
                ArrayPool<float>.Shared.Return(buffer, clearArray: true);
            }

            RaiseErrorAsync(exception, "Capture callback failed.");
        }
    }

    private static int RoundUpToPowerOf2(int value)
    {
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        return value + 1;
    }

    // Overflow keeps the newest audio: the oldest queued block is dropped and returned to the
    // pool, so a consumer that falls behind cannot deliver stale audio while newer audio is
    // discarded.
    private static Channel<CaptureBuffer> CreateCaptureChannel(uint capacity)
        => Channel.CreateBounded<CaptureBuffer>(
            new BoundedChannelOptions((int)capacity)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true,
            },
            static dropped => dropped.Release());

    private async Task ProcessQueueAsync(Channel<CaptureBuffer> channel)
    {
        try
        {
            await foreach (CaptureBuffer item in channel.Reader.ReadAllAsync().ConfigureAwait(false))
            {
                try
                {
                    MiniAudioServiceHelpers.InvokeDataHandlers(
                        OnDataAvailable,
                        item.Buffer.AsSpan(0, item.Length));
                }
                finally
                {
                    item.Release();
                }
            }
        }
        catch (Exception exception)
        {
            RaiseError(exception, "Capture processing queue failed.");
        }
    }

    private async Task DrainPipelineAsync()
    {
        Task? processingTask = _processingTask;
        _processingTask = null;
        if (processingTask is not null)
        {
            await processingTask.ConfigureAwait(false);
        }

        Channel<CaptureBuffer>? channel = _captureChannel;
        if (channel is null)
        {
            return;
        }

        while (channel.Reader.TryRead(out CaptureBuffer item))
        {
            item.Release();
        }
    }

    private void RaiseError(Exception exception, string message)
        => MiniAudioServiceHelpers.InvokeEventHandlers(
            OnError,
            this,
            new AudioErrorEventArgs(exception, message, _state));

    private void RaiseErrorAsync(Exception exception, string message)
        => _ = Task.Run(() => RaiseError(exception, message));

    // Ownership transfers to the channel only after a successful TryWrite.
    private readonly struct CaptureBuffer(float[] buffer, int length)
    {
        internal float[] Buffer { get; } = buffer;

        internal int Length { get; } = length;

        internal void Release()
        {
            ArrayPool<float>.Shared.Return(Buffer, clearArray: true);
        }
    }
}
