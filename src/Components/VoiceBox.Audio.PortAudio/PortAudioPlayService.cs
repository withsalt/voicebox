using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Options;
using SharpASR.AudioService.Backend;
using VoiceBox.Audio.PortAudio.Backend.Enums;
using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio
{
    public sealed class PortAudioPlayService : PortAudioServiceBase, IAudioPlayService
    {
        private readonly IOptions<PortAudioPlayOptions> _options;
        private PortAudioStream? _stream;
        private volatile bool _isPlaying = false;
        private int _currentDeviceIndex = -1;

        private QueueBuffer? _currentBuffer = null;
        private int _currentBufferPosition = 0;
        private readonly Channel<QueueBuffer> _channel;

        private readonly ManualResetEventSlim _audioDataRequestedEvent = new ManualResetEventSlim(true);
        private readonly ManualResetEventSlim _resetEvent = new ManualResetEventSlim(true);

        /// <summary>
        /// 播放完成事件（队列为空且播放停止）
        /// </summary>
        public event EventHandler? OnPlaybackCompleted;

        /// <summary>
        /// 队列溢出事件（数据被丢弃）
        /// </summary>
        public event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow;

        /// <summary>
        /// 错误事件
        /// </summary>
        public event EventHandler<AudioErrorEventArgs>? OnError;

        /// <summary>
        /// Xrun 事件（音频缓冲区下溢/上溢）
        /// </summary>
        public event EventHandler<AudioXrunEventArgs>? OnXrun;

        public PortAudioPlayService(IOptions<PortAudioPlayOptions> options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));

            OptionsValidator(_options.Value);

            _channel = Channel.CreateBounded<QueueBuffer>(new BoundedChannelOptions((int)_options.Value.MaxQueueSize)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false
            });

            AcquirePortAudioReference();
        }

        public bool IsPlaying => _isPlaying;

        public override AudioOptionsBase Options => _options.Value;

        public int CurrentQueueSize
        {
            get
            {
                return _channel.Reader.Count;
            }
        }

        public int GetDefaultOutputDevice()
        {
            CheckDisposed();

            return Backend.PortAudioBackend.DefaultOutputDevice;
        }

        public async Task StartPlayingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                await StartPlayingInternalAsync(deviceIndex, cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private async Task StartPlayingInternalAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
        {
            if (_state == AudioServiceState.Active)
            {
                return; // Already started
            }

            if (_state == AudioServiceState.Initializing || _state == AudioServiceState.Stopping)
            {
                throw new InvalidOperationException($"Cannot start playing while in {_state} state.");
            }

            _state = AudioServiceState.Initializing;

            nint wasapiStreamInfo = nint.Zero;
            try
            {
                // Wait for any pending reset operation
                if (!_resetEvent.Wait(3000))
                {
                    throw new TimeoutException("Audio queue is being cleaned, cannot start playing at this moment.");
                }

                    CleanupQueue();

                    DeviceInfo deviceInfo = GetAudioDevice(ref deviceIndex, AudioDeviceType.Output);
                    _currentDeviceIndex = deviceIndex;

                    double suggestedLatency = _options.Value.SuggestedLatency ?? deviceInfo.defaultLowOutputLatency;

                    // WASAPI 设备需要专属 stream info: 共享模式带 AutoConvert
                    // (允许与系统混音格式不一致的流), 独占模式按选项启用
                    wasapiStreamInfo = Backend.PortAudioBackend.AllocWasapiStreamInfo(deviceIndex, _options.Value.EnableWasapiExclusiveMode);

                    StreamParameters param = new StreamParameters
                    {
                        device = deviceIndex,
                        channelCount = (int)_options.Value.ChannelCount,
                        sampleFormat = _options.Value.SampleFormat,
                        suggestedLatency = suggestedLatency,
                        hostApiSpecificStreamInfo = wasapiStreamInfo
                    };

                    uint frameSize = CalculateFrameSize(deviceInfo);

                    _stream = new PortAudioStream(null, param, _options.Value.SampleRate, frameSize, StreamFlags.ClipOff, OnAudioDataRequested, nint.Zero);
                    _stream.Start();

                    // Verify stream is active
                    await Task.Delay(50, cancellationToken); // Give it a moment to start
                    if (!_stream.IsActive)
                    {
                        throw new InvalidOperationException("Failed to start audio stream.");
                    }

                    _state = AudioServiceState.Active;
                    _isPlaying = false;
                }
                catch (Exception ex)
                {
                    _state = AudioServiceState.Error;
                    CleanupStream();
                    OnError?.Invoke(this, new AudioErrorEventArgs(ex, "Failed to start playing", AudioServiceState.Error));
                    throw;
                }
                finally
                {
                    // Pa_OpenStream 打开时已把 stream info 拷贝进流自身状态, 此处统一释放
                    Backend.PortAudioBackend.FreeWasapiStreamInfo(wasapiStreamInfo);
                }
        }

        private uint CalculateFrameSize(DeviceInfo deviceInfo)
        {
            if (_options.Value.MaxFrameSize == -1)
                return (uint)Math.Max(0.0, deviceInfo.defaultLowOutputLatency * _options.Value.SampleRate);
            else if (_options.Value.MaxFrameSize == 0)
                return 0;
            else
                return (uint)_options.Value.MaxFrameSize;
        }

        public async Task StopPlayingAsync(CancellationToken cancellationToken = default)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                await StopPlayingInternalAsync(cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private async Task StopPlayingInternalAsync(CancellationToken cancellationToken = default)
        {
            if (_state == AudioServiceState.Idle)
            {
                return; // Already stopped
            }

            if (_state == AudioServiceState.Stopping)
            {
                return; // Already stopping
            }

            _state = AudioServiceState.Stopping;

            try
            {
                CleanupStream();
                    _state = AudioServiceState.Idle;
                    _isPlaying = false;
                    _currentDeviceIndex = -1;
                }
                catch (Exception ex)
                {
                    _state = AudioServiceState.Error;
                    OnError?.Invoke(this, new AudioErrorEventArgs(ex, "Failed to stop playing", AudioServiceState.Error));
                    throw;
                }
            // Add await to suppress async warning
            await Task.CompletedTask;
        }

        /// <summary>
        /// 尝试向播放队列添加音频数据
        /// </summary>
        /// <param name="data">音频数据</param>
        /// <param name="droppedSamples">如果队列满，返回被丢弃的样本数</param>
        /// <returns>是否成功入队</returns>
        public bool TryEnqueue(ReadOnlySpan<float> data, out int droppedSamples)
        {
            droppedSamples = 0;
            CheckDisposed();

            if (!_resetEvent.Wait(3000))
            {
                droppedSamples = data.Length;
                return false;
            }

            if (data.IsEmpty)
                return true;

            if (_state != AudioServiceState.Active)
            {
                droppedSamples = data.Length;
                return false;
            }

            var buffer = ArrayPool<float>.Shared.Rent(data.Length);
            try
            {
                data.CopyTo(buffer);
                if (!_channel.Writer.TryWrite(new QueueBuffer(buffer, data.Length)))
                {
                    ArrayPool<float>.Shared.Return(buffer);
                    droppedSamples = data.Length;

                    // Trigger overflow event
                    OnQueueOverflow?.Invoke(this, new AudioQueueOverflowEventArgs(
                        droppedSamples,
                        _channel.Reader.Count,
                        (int)_options.Value.MaxQueueSize));

                    Debug.WriteLine($"[Warning]Audio queue overflow: {droppedSamples} samples discarded.");
                    return false;
                }
                return true;
            }
            catch
            {
                ArrayPool<float>.Shared.Return(buffer, true);
                droppedSamples = data.Length;
                throw;
            }
        }

        /// <summary>
        /// 异步切换播放设备
        /// </summary>
        public async Task SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken = default)
        {
            CheckDisposed();

            if (deviceIndex == _currentDeviceIndex)
            {
                return; // Already using this device
            }

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                if (_state != AudioServiceState.Active)
                {
                    throw new InvalidOperationException("Cannot switch device when not playing. Start playing first.");
                }

                int oldDeviceIndex = _currentDeviceIndex;

                // Stop current stream (internal, no lock)
                await StopPlayingInternalAsync(cancellationToken);

                // Start with new device (internal, no lock)
                await StartPlayingInternalAsync(deviceIndex, cancellationToken);

                Debug.WriteLine($"[Info] Switched audio device from {oldDeviceIndex} to {deviceIndex}");
            }
            catch (Exception ex)
            {
                OnError?.Invoke(this, new AudioErrorEventArgs(ex, $"Failed to switch device to {deviceIndex}", _state));
                throw;
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        /// <summary>
        /// 清空播放队列
        /// </summary>
        public void ClearQueue()
        {
            CheckDisposed();

            _asyncLock.Wait();
            try
            {
                _resetEvent.Reset();
                try
                {
                    if (_stream != null && _stream.IsActive)
                    {
                        _stream.Stop();
                        _audioDataRequestedEvent.Wait(1000);
                    }

                    CleanupQueue();

                    if (_isPlaying)
                        _isPlaying = false;

                    if (_stream != null)
                        _stream.Start();
                }
                finally
                {
                    _resetEvent.Set();
                }
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private unsafe StreamCallbackResult OnAudioDataRequested(nint input, nint output, uint frameCount, ref StreamCallbackTimeInfo timeInfo, StreamCallbackFlags statusFlags, nint userData)
        {
            _audioDataRequestedEvent.Reset();

            // 检测 xrun（输出下溢）
            if ((statusFlags & StreamCallbackFlags.OutputUnderflow) != 0)
            {
                Task.Run(() => OnXrun?.Invoke(this, new AudioXrunEventArgs(AudioXrunType.OutputUnderflow, frameCount)));
                Debug.WriteLine("[Warning] Output underflow detected (xrun)");
            }

            try
            {
                int totalSamples = (int)frameCount * (int)_options.Value.ChannelCount;
                float* outputPtr = (float*)output.ToPointer();
                int samplesWritten = 0;

                if (!_resetEvent.IsSet)
                {
                    var remainingBuffer = new Span<float>(&outputPtr[samplesWritten], totalSamples - samplesWritten);
                    remainingBuffer.Clear();
                    return StreamCallbackResult.Continue;
                }

                while (samplesWritten < totalSamples && _resetEvent.IsSet)
                {
                    // 1. 归还旧缓冲区
                    if (_currentBuffer != null && _currentBufferPosition >= _currentBuffer.Length)
                    {
                        _currentBuffer.Release();
                        _currentBuffer = null;
                        _currentBufferPosition = 0;
                    }

                    // 2. 尝试从队列获取新缓冲区
                    if (_currentBuffer == null)
                    {
                        if (_channel.Reader.TryRead(out var newBuffer))
                        {
                            _currentBuffer = newBuffer;
                            _currentBufferPosition = 0;

                            if (!_isPlaying) _isPlaying = true;
                        }
                        else
                        {
                            if (_isPlaying)
                            {
                                _isPlaying = false;
                                // Trigger playback completed event on transition from playing to not playing
                                Task.Run(() => OnPlaybackCompleted?.Invoke(this, EventArgs.Empty));
                            }
                            break;
                        }
                    }

                    // 3. 从当前缓冲区复制数据
                    int samplesAvailable = !_resetEvent.IsSet ? 0 : _currentBuffer.Length - _currentBufferPosition;
                    int samplesToCopy = Math.Min(samplesAvailable, totalSamples - samplesWritten);
                    if (samplesToCopy > 0)
                    {
                        var sourceSpan = new Span<float>(_currentBuffer.Data, _currentBufferPosition, samplesToCopy);
                        var destinationSpan = new Span<float>(&outputPtr[samplesWritten], samplesToCopy);
                        sourceSpan.CopyTo(destinationSpan);

                        samplesWritten += samplesToCopy;
                        _currentBufferPosition += samplesToCopy;
                    }
                }

                // 3. 如果缓冲区未填满，用静音填充剩余部分
                if (samplesWritten < totalSamples)
                {
                    var remainingBuffer = new Span<float>(&outputPtr[samplesWritten], totalSamples - samplesWritten);
                    remainingBuffer.Clear();
                }

                // 如果流没有被外部停止，就继续运行（即使当前是静音）
                // 只有在数据源明确表示结束时才返回 Complete
                return StreamCallbackResult.Continue;
            }
            finally
            {
                _audioDataRequestedEvent.Set();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CleanupStream()
        {
            PortAudioStream? stream = _stream;
            _stream = null;

            if (stream != null)
            {
                try
                {
                    if (stream.IsActive)
                    {
                        stream.Stop();
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[ERROR] Failed to stop PortAudio stream: {ex.Message}\r\n{ex.StackTrace}");
                    try
                    {
                        stream.Abort();
                    }
                    catch (Exception abortEx)
                    {
                        Debug.WriteLine($"[ERROR] Failed to abort PortAudio stream: {abortEx.Message}\r\n{abortEx.StackTrace}");
                    }
                }

                // 等待回调完成，避免资源冲突
                _audioDataRequestedEvent.Wait(1000);

                try
                {
                    stream.Close();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[ERROR] Failed to close PortAudio stream: {ex.Message}\r\n{ex.StackTrace}");
                }

                try
                {
                    stream.Dispose();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[ERROR] Failed to dispose PortAudio stream: {ex.Message}\r\n{ex.StackTrace}");
                }
            }

            CleanupQueue();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CleanupQueue()
        {
            try
            {
                // 清理队列
                while (_channel.Reader.TryRead(out var buffer))
                {
                    buffer.Release();
                }
                // 清理当前缓冲区
                if (_currentBuffer != null)
                {
                    _currentBuffer.Release();
                    _currentBuffer = null;
                    _currentBufferPosition = 0;
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ERROR] Failed to clean up queue: {ex.Message}\r\n{ex.StackTrace}");
            }
        }

        protected override void OptionsValidator(AudioOptionsBase options)
        {
            base.OptionsValidator(options);

            if (_options.Value.MaxFrameSize < -1)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.MaxFrameSize), "MaxFrameSize must be -1 (auto), 0 (PortAudio decides), or a positive integer.");

            if (_options.Value.MaxQueueSize < 10 || _options.Value.MaxQueueSize > 10000)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.MaxQueueSize), $"Max queue size must be between 10-10000. Current value: {_options.Value.MaxQueueSize}");

            if (_options.Value.SuggestedLatency.HasValue && (_options.Value.SuggestedLatency.Value < 0.001 || _options.Value.SuggestedLatency.Value > 1.0))
                throw new ArgumentOutOfRangeException(nameof(_options.Value.SuggestedLatency), "SuggestedLatency must be between 0.001 and 1.0 seconds.");
        }

        public override void Dispose()
        {
            if (_disposed)
                return;

            _channel.Writer.TryComplete();

            try
            {
                StopPlayingAsync().GetAwaiter().GetResult();
            }
            catch
            {
                // Ignore errors during disposal
            }

            _audioDataRequestedEvent?.Dispose();
            _resetEvent?.Dispose();

            GC.SuppressFinalize(this);
            base.Dispose();
        }
    }
}
