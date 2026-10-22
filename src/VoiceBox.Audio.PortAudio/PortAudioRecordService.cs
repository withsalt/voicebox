using System.Buffers;
using System.Diagnostics;
using System.Threading.Channels;
using Microsoft.Extensions.Options;
using SharpASR.AudioService.Backend;
using VoiceBox.Audio.PortAudio.Backend;
using VoiceBox.Audio.PortAudio.Backend.Enums;
using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Audio.PortAudio
{
    public sealed class PortAudioRecordService : PortAudioServiceBase, IAudioRecordService
    {
        private readonly IOptions<PortAudioRecordOptions> _options;
        private PortAudioStream? _stream;
        private volatile bool _isRecording = false;
        private int _currentDeviceIndex = -1;

        public event Action<ReadOnlySpan<float>>? OnDataAvailable;

        /// <summary>
        /// 错误事件
        /// </summary>
        public event EventHandler<AudioErrorEventArgs>? OnError;

        /// <summary>
        /// Xrun 事件（音频缓冲区下溢/上溢）
        /// </summary>
        public event EventHandler<AudioXrunEventArgs>? OnXrun;

        private Channel<(float[] Buffer, int Length)>? _dataChannel;
        private Task? _processingTask;
        private CancellationTokenSource? _cancellationTokenSource;

        public bool IsRecording => _isRecording;

        public override AudioOptionsBase Options => _options.Value;

        public PortAudioRecordService(IOptions<PortAudioRecordOptions> options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));

            OptionsValidator(_options.Value);
            AcquirePortAudioReference();
        }

        public int GetDefaultInputDevice()
        {
            return Backend.PortAudioBackend.DefaultInputDevice;
        }

        public async Task StartRecordingAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                await StartRecordingInternalAsync(deviceIndex, cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private async Task StartRecordingInternalAsync(int deviceIndex = -1, CancellationToken cancellationToken = default)
        {
            if (_state == AudioServiceState.Active)
            {
                return; // Already started
            }

            if (_state == AudioServiceState.Initializing || _state == AudioServiceState.Stopping)
            {
                throw new InvalidOperationException($"Cannot start recording while in {_state} state.");
            }

            if (_options.Value.SampleFormat != SampleFormat.Float32)
            {
                throw new NotSupportedException("Only SampleFormat.Float32 is supported.");
            }

            _state = AudioServiceState.Initializing;

            nint wasapiStreamInfo = nint.Zero;
            try
            {
                DeviceInfo deviceInfo = GetAudioDevice(ref deviceIndex, AudioDeviceType.Input);
                _currentDeviceIndex = deviceIndex;

                    double suggestedLatency = _options.Value.SuggestedLatency ?? deviceInfo.defaultLowInputLatency;

                    // WASAPI 设备需要专属 stream info: 共享模式带 AutoConvert
                    // (允许 16kHz 等与系统混音格式不一致的流), 独占模式按选项启用
                    wasapiStreamInfo = Backend.PortAudioBackend.AllocWasapiStreamInfo(deviceIndex, _options.Value.EnableWasapiExclusiveMode);

                    var streamParameters = new StreamParameters
                    {
                        device = deviceIndex,
                        channelCount = (int)_options.Value.ChannelCount,
                        sampleFormat = _options.Value.SampleFormat,
                        suggestedLatency = suggestedLatency,
                        hostApiSpecificStreamInfo = wasapiStreamInfo
                    };

                    double sampleRate = _options.Value.SampleRate;
                    if (sampleRate <= 0)
                    {
                        throw new ArgumentOutOfRangeException(nameof(_options.Value.SampleRate), "Sample rate must be greater than zero.");
                    }

                    if (_options.Value.ChannelCount > deviceInfo.maxInputChannels)
                    {
                        throw new ArgumentOutOfRangeException(nameof(_options.Value.ChannelCount), $"Channel count {_options.Value.ChannelCount} exceeds maximum input channels {deviceInfo.maxInputChannels} for device {deviceInfo.name}.");
                    }

                    var processingCts = new CancellationTokenSource();
                    _cancellationTokenSource = processingCts;

                    var dataChannel = RealtimeAudioChannel.CreateDropOldest<(float[] Buffer, int Length)>(
                        (int)_options.Value.BufferCapacity,
                        singleReader: true,
                        singleWriter: true,
                        static dropped => ArrayPool<float>.Shared.Return(dropped.Buffer, true));
                    _dataChannel = dataChannel;

                    var taskStartedSignal = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                    _processingTask = Task.Factory.StartNew(
                        () =>
                        {
                            taskStartedSignal.TrySetResult(true);
                            ProcessQueue(dataChannel, processingCts.Token);
                        },
                        CancellationToken.None,
                        TaskCreationOptions.LongRunning,
                        TaskScheduler.Default);

                    // Wait for task to start (async, no polling)
                    await taskStartedSignal.Task.WaitAsync(TimeSpan.FromSeconds(1), cancellationToken);

                    uint frameSize = CalculateFrameSize(deviceInfo, sampleRate);
                    _stream = new PortAudioStream(
                        inParams: streamParameters,
                        outParams: null,
                        sampleRate: sampleRate,
                        framesPerBuffer: frameSize,
                        streamFlags: StreamFlags.ClipOff,
                        callback: OnAudioDataReceived,
                        userData: nint.Zero
                    );

                    _stream.Start();

                    // Verify stream is active
                    await Task.Delay(50, cancellationToken);
                    if (!_stream.IsActive)
                    {
                        throw new InvalidOperationException("Failed to start audio stream.");
                    }

                    _isRecording = true;
                    _state = AudioServiceState.Active;
                }
                catch (Exception ex)
                {
                    _state = AudioServiceState.Error;

                    _isRecording = false;
                    Exception? pipelineError = await CleanupRecordPipelineAsync(CancellationToken.None);
                    if (pipelineError != null)
                    {
                        Debug.WriteLine($"[ERROR] Audio processing task failed during start cleanup: {pipelineError.Message}\r\n{pipelineError.StackTrace}");
                    }

                    Exception? cleanupError = CleanupStream();
                    if (cleanupError != null)
                    {
                        Debug.WriteLine($"[ERROR] Failed to clean up recording stream after start failure: {cleanupError.Message}\r\n{cleanupError.StackTrace}");
                    }

                    OnError?.Invoke(this, new AudioErrorEventArgs(ex, "Failed to start recording", AudioServiceState.Error));
                    throw;
                }
                finally
                {
                    // Pa_OpenStream 打开时已把 stream info 拷贝进流自身状态, 此处统一释放
                    Backend.PortAudioBackend.FreeWasapiStreamInfo(wasapiStreamInfo);
                }
        }

        public async Task StopRecordingAsync(CancellationToken cancellationToken = default)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                await StopRecordingInternalAsync(cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private async Task StopRecordingInternalAsync(CancellationToken cancellationToken = default)
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
                _isRecording = false;

                    Exception? pipelineError = await CleanupRecordPipelineAsync(cancellationToken);

                    Exception? cleanupError = CleanupStream();
                    if (cleanupError != null)
                    {
                        throw cleanupError;
                    }

                    if (pipelineError != null)
                    {
                        throw pipelineError;
                    }

                    _state = AudioServiceState.Idle;
                    _currentDeviceIndex = -1;
                }
                catch (Exception ex)
                {
                    _state = AudioServiceState.Error;
                    OnError?.Invoke(this, new AudioErrorEventArgs(ex, "Failed to stop recording", AudioServiceState.Error));
                    throw;
                }
        }

        private unsafe StreamCallbackResult OnAudioDataReceived(nint input, nint output, uint frameCount, ref StreamCallbackTimeInfo timeInfo, StreamCallbackFlags statusFlags, nint userData)
        {
            // 检测 xrun（输入溢出）
            if ((statusFlags & StreamCallbackFlags.InputOverflow) != 0)
            {
                Task.Run(() => OnXrun?.Invoke(this, new AudioXrunEventArgs(AudioXrunType.InputOverflow, frameCount)));
                Debug.WriteLine("[Warning] Input overflow detected (xrun)");
            }

            // Snapshot the channel/cts references before any null check so the stop path
            // (which assigns null to these fields) can't race the body of this callback.
            var dataChannel = _dataChannel;
            var cts = _cancellationTokenSource;

            if (!_isRecording
                || OnDataAvailable == null
                || frameCount == 0
                || dataChannel == null
                || cts?.IsCancellationRequested == true)
            {
                return StreamCallbackResult.Continue;
            }

            int totalSamples = (int)(frameCount * _options.Value.ChannelCount);
            if (totalSamples == 0)
            {
                return StreamCallbackResult.Continue;
            }

            float[] buffer = ArrayPool<float>.Shared.Rent(totalSamples);
            var inputSpan = new ReadOnlySpan<float>(input.ToPointer(), totalSamples);
            inputSpan.CopyTo(buffer);

            if (!dataChannel.Writer.TryWrite((buffer, totalSamples)))
            {
                ArrayPool<float>.Shared.Return(buffer, true);
                Debug.WriteLine("[Warning]Failed to write audio data to channel. Channel may be full or closed.");
            }

            return StreamCallbackResult.Continue;
        }

        /// <summary>
        /// 异步切换录音设备
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
                    throw new InvalidOperationException("Cannot switch device when not recording. Start recording first.");
                }

                int oldDeviceIndex = _currentDeviceIndex;

                // Stop current stream (internal, no lock)
                await StopRecordingInternalAsync(cancellationToken);

                // Start with new device (internal, no lock)
                await StartRecordingInternalAsync(deviceIndex, cancellationToken);

                Debug.WriteLine($"[Info] Switched recording device from {oldDeviceIndex} to {deviceIndex}");
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

        private void ProcessQueue(Channel<(float[] Buffer, int Length)> dataChannel, CancellationToken token)
        {
            using var mmcss = MmcssThreadScope.TryEnter("Pro Audio");
            TrySetCurrentThreadPriority(ThreadPriority.AboveNormal);
            try
            {
                while (RealtimeAudioChannel.WaitToReadBlocking(dataChannel.Reader, token))
                {
                    while (dataChannel.Reader.TryRead(out var item))
                    {
                        try
                        {
                            if (dataChannel.Reader.Count >= 5)
                            {
                                Debug.WriteLine($"[Warning]Audio data queue is over {dataChannel.Reader.Count} items. Please consider improving the processing performance.");
                            }

                            var memory = new ReadOnlySpan<float>(item.Buffer, 0, item.Length);
                            OnDataAvailable?.Invoke(memory);
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(item.Buffer, true);
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine("Audio processing task was canceled.");
            }
        }

        private static void TrySetCurrentThreadPriority(ThreadPriority priority)
        {
            try
            {
                Thread.CurrentThread.Priority = priority;
            }
            catch (PlatformNotSupportedException)
            {
                Debug.WriteLine("[Info] Thread priority is not supported on this platform (record processing).");
            }
        }

        private async Task<Exception?> CleanupRecordPipelineAsync(CancellationToken cancellationToken)
        {
            var dataChannel = _dataChannel;
            var cts = _cancellationTokenSource;
            var processingTask = _processingTask;
            Exception? processingError = null;

            dataChannel?.Writer.TryComplete();
            bool processingTaskCompleted = processingTask == null;

            if (processingTask != null)
            {
                try
                {
                    await processingTask.WaitAsync(TimeSpan.FromSeconds(2));
                    processingTaskCompleted = true;
                }
                catch (OperationCanceledException)
                {
                    processingTaskCompleted = true;
                }
                catch (TimeoutException)
                {
                    try
                    {
                        cts?.Cancel();
                    }
                    catch (ObjectDisposedException)
                    {
                    }

                    try
                    {
                        await processingTask.WaitAsync(TimeSpan.FromSeconds(1));
                        processingTaskCompleted = true;
                    }
                    catch (OperationCanceledException)
                    {
                        processingTaskCompleted = true;
                    }
                    catch (TimeoutException)
                    {
                    }
                    catch (Exception ex)
                    {
                        processingTaskCompleted = true;
                        processingError = ex;
                        Debug.WriteLine($"[ERROR] Audio processing task failed during cleanup: {ex.Message}\r\n{ex.StackTrace}");
                    }
                }
                catch (Exception ex)
                {
                    processingTaskCompleted = true;
                    processingError = ex;
                    Debug.WriteLine($"[ERROR] Audio processing task failed during cleanup: {ex.Message}\r\n{ex.StackTrace}");
                }
            }

            if (processingTaskCompleted)
            {
                DrainRecordChannel(dataChannel);
                cts?.Dispose();
            }
            else
            {
                _ = processingTask!.ContinueWith(
                    _ =>
                    {
                        DrainRecordChannel(dataChannel);
                        cts?.Dispose();
                    },
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
            }

            _cancellationTokenSource = null;
            _processingTask = null;
            _dataChannel = null;

            return processingError;
        }

        private static void DrainRecordChannel(Channel<(float[] Buffer, int Length)>? dataChannel)
        {
            if (dataChannel == null)
                return;

            while (dataChannel.Reader.TryRead(out var item))
            {
                ArrayPool<float>.Shared.Return(item.Buffer, true);
            }
        }

        private Exception? CleanupStream()
        {
            PortAudioStream? stream = _stream;
            _stream = null;

            if (stream == null)
                return null;

            Exception? firstError = null;

            try
            {
                if (stream.IsActive)
                {
                    stream.Stop();
                }
            }
            catch (Exception ex)
            {
                firstError ??= ex;
                Debug.WriteLine($"[ERROR] Failed to stop PortAudio recording stream: {ex.Message}\r\n{ex.StackTrace}");
                try
                {
                    stream.Abort();
                }
                catch (Exception abortEx)
                {
                    firstError ??= abortEx;
                    Debug.WriteLine($"[ERROR] Failed to abort PortAudio recording stream: {abortEx.Message}\r\n{abortEx.StackTrace}");
                }
            }

            try
            {
                stream.Close();
            }
            catch (Exception ex)
            {
                firstError ??= ex;
                Debug.WriteLine($"[ERROR] Failed to close PortAudio recording stream: {ex.Message}\r\n{ex.StackTrace}");
            }

            try
            {
                stream.Dispose();
            }
            catch (Exception ex)
            {
                firstError ??= ex;
                Debug.WriteLine($"[ERROR] Failed to dispose PortAudio recording stream: {ex.Message}\r\n{ex.StackTrace}");
            }

            return firstError;
        }

        private uint CalculateFrameSize(DeviceInfo deviceInfo, double sampleRate)
        {
            if (_options.Value.MaxFrameSize == -1)
                return (uint)Math.Max(0.0, (_options.Value.SuggestedLatency ?? deviceInfo.defaultLowInputLatency) * sampleRate);
            else if (_options.Value.MaxFrameSize == 0)
                return 0;
            else
                return (uint)_options.Value.MaxFrameSize;
        }

        protected override void OptionsValidator(AudioOptionsBase options)
        {
            base.OptionsValidator(options);

            if (_options.Value.BufferCapacity < 1 || _options.Value.BufferCapacity > short.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.BufferCapacity), $"Max buffer capacity size must be between 1-{short.MaxValue}. Current value: {_options.Value.BufferCapacity}");

            if (_options.Value.MaxFrameSize < -1)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.MaxFrameSize), "MaxFrameSize must be -1 (auto), 0 (PortAudio decides), or a positive integer.");

            if (_options.Value.SuggestedLatency.HasValue && (_options.Value.SuggestedLatency.Value < 0.001 || _options.Value.SuggestedLatency.Value > 1.0))
                throw new ArgumentOutOfRangeException(nameof(_options.Value.SuggestedLatency), "SuggestedLatency must be between 0.001 and 1.0 seconds.");
        }

        public override void Dispose()
        {
            if (_disposed)
                return;

            try
            {
                StopRecordingAsync().GetAwaiter().GetResult();
            }
            catch
            {
                // Ignore errors during disposal
            }

            GC.SuppressFinalize(this);
            base.Dispose();
        }
    }
}
