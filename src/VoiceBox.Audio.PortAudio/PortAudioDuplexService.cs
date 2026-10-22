using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Options;
using SharpASR.AudioService.Backend;
using VoiceBox.Audio.PortAudio.Backend.Enums;
using VoiceBox.Audio.PortAudio.Backend.Models;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.Core.Services.HPFService;

namespace VoiceBox.Audio.PortAudio
{
    /// <summary>
    /// 全双工音频服务：同时支持录音和播放，使用单个 PortAudio stream。
    /// 相比分离的录音和播放服务，全双工模式具有更低的延迟和更好的同步性。
    /// </summary>
    public sealed class PortAudioDuplexService : PortAudioServiceBase, IAudioDuplexService
    {
        private readonly IOptions<PortAudioDuplexOptions> _options;
        private PortAudioStream? _stream;
        private volatile bool _isRecording = false;
        private volatile bool _isPlaying = false;
        private int _currentInputDeviceIndex = -1;
        private int _currentOutputDeviceIndex = -1;

        // 录音相关
        private Channel<RecordAudioBuffer>? _recordChannel;
        private Task? _recordProcessingTask;
        private Channel<DuplexAudioFrameBuffer>? _duplexFrameChannel;
        private Task? _duplexFrameDispatchTask;
        private CancellationTokenSource? _cancellationTokenSource;
        private long _streamEpochCounter;
        private long _captureStreamEpoch;
        private long _captureSequence;
        private long _captureSamplePosition;
        private long _renderStreamEpoch;
        private long _renderSamplePosition;
        private readonly BufferedCaptureTimelineMapper _captureTimelineMapper = new();
        private IAcousticEchoCancellationService? _captureTimelineProcessor;
        private bool _captureTimelineProcessorInitialized;

        // 播放相关
        private QueueBuffer? _currentBuffer = null;
        private int _currentBufferPosition = 0;
        private readonly Channel<QueueBuffer> _playChannel;
        private readonly ManualResetEventSlim _audioDataRequestedEvent = new ManualResetEventSlim(true);
        private readonly ManualResetEventSlim _resetEvent = new ManualResetEventSlim(true);

        // 播放增益 (用于 barge-in ducking)：控制线程设目标增益，实时回调内按斜坡平滑逼近，避免增益突变产生咔哒声。
        private const float PlaybackGainRampMs = 15f;
        private float _targetPlaybackGain = 1.0f;   // 控制线程写 / 回调线程读 (Volatile)
        private float _currentPlaybackGain = 1.0f;   // 仅回调线程访问 (及 ClearQueue 停流窗口)

        // 事件
        public event Action<ReadOnlySpan<float>>? OnDataAvailable;
        public event EventHandler? OnPlaybackCompleted;
        public event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow;
        public event EventHandler<AudioErrorEventArgs>? OnError;
        public event EventHandler<AudioXrunEventArgs>? OnXrun;
        public event DuplexAudioFrameHandler? OnDuplexAudioFrame;
        public event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;

        public override AudioOptionsBase Options => _options.Value;

        public bool IsRecording => _isRecording;
        public bool IsPlaying => _isPlaying;

        public int CurrentQueueSize => _playChannel.Reader.Count;

        /// <summary>
        /// 可选的采集处理器：在全双工回调内对麦克风信号做处理（典型为回声消除 AEC），
        /// 可访问与采集时间对齐的播放(远端)参考。设置后还需 <see cref="_enableAec"/> 为 true 才生效。
        /// </summary>
        public IAcousticEchoCancellationService? _aecService { get; set; }

        /// <summary>
        /// 可选的录音降噪处理器：在录音队列派发前对麦克风信号做流式降噪。
        /// 设置后还需 <see cref="_enableDnr"/> 为 true 才生效。
        /// </summary>
        public IDigitalNoiseReductionService? _dnrService { get; set; }

        /// <summary>
        /// 可选的高通滤波处理器：在全双工回调内对麦克风信号做高通滤波，
        /// 置于采集处理链最前端（AEC/DNR 之前）。设置后还需 <see cref="_enableHpf"/> 为 true 才生效。
        /// </summary>
        public IHighPassFilterService? _hpfService { get; set; }

        /// <summary>
        /// 采集处理器开关（运行时可切换，A/B 对比）。跨线程仅作 bool 读写。
        /// </summary>
        private volatile bool _enableAec;

        /// <summary>
        /// 降噪处理器开关（运行时可切换，A/B 对比）。跨线程仅作 bool 读写。
        /// </summary>
        private volatile bool _enableDnr;

        /// <summary>
        /// 高通滤波处理器开关（运行时可切换，A/B 对比）。跨线程仅作 bool 读写。
        /// </summary>
        private volatile bool _enableHpf;

        public bool EnableAec { get => _enableAec; set => _enableAec = value; }
        public bool EnableDnr { get => _enableDnr; set => _enableDnr = value; }
        public bool EnableHighPassFilter { get => _enableHpf; set => _enableHpf = value; }

        public PortAudioDuplexService(
            IOptions<PortAudioDuplexOptions> options,
            IAcousticEchoCancellationService? aecService = null,
            IDigitalNoiseReductionService? dnrService = null,
            IHighPassFilterService? hpfService = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            OptionsValidator(_options.Value);

            _aecService = aecService;
            _dnrService = dnrService;
            _hpfService = hpfService;

            if (_options.Value.EnableAec)
            {
                if (_aecService == null)
                    throw new InvalidOperationException("AEC is enabled in PortAudioFullDuplexOptions, but no IAcousticEchoCancellationService is registered.");
                _enableAec = true;
            }

            if (_options.Value.EnableDnr)
            {
                if (_dnrService == null)
                    throw new InvalidOperationException("DNR is enabled in PortAudioFullDuplexOptions, but no IDigitalNoiseReductionService is registered.");
                _enableDnr = true;
            }

            if (_options.Value.EnableHpf)
            {
                if (_hpfService == null)
                    throw new InvalidOperationException("High-pass filter is enabled in PortAudioFullDuplexOptions, but no IHighPassFilterService is registered.");
                _enableHpf = true;
            }

            _playChannel = Channel.CreateBounded<QueueBuffer>(new BoundedChannelOptions((int)_options.Value.PlayMaxQueueSize)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false
            });

            AcquirePortAudioReference();
        }

        public int GetDefaultInputDevice()
        {
            CheckDisposed();
            return Backend.PortAudioBackend.DefaultInputDevice;
        }

        public int GetDefaultOutputDevice()
        {
            CheckDisposed();
            return Backend.PortAudioBackend.DefaultOutputDevice;
        }

        /// <summary>
        /// 启动全双工音频流（同时录音和播放）
        /// </summary>
        public async Task StartAsync(int inputDeviceIndex = -1, int outputDeviceIndex = -1, CancellationToken cancellationToken = default)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                await StartInternalAsync(inputDeviceIndex, outputDeviceIndex, cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private async Task StartInternalAsync(int inputDeviceIndex = -1, int outputDeviceIndex = -1, CancellationToken cancellationToken = default)
        {
            if (_state == AudioServiceState.Active)
            {
                return; // Already started
            }

            if (_state == AudioServiceState.Initializing || _state == AudioServiceState.Stopping)
            {
                throw new InvalidOperationException($"Cannot start while in {_state} state.");
            }

            if (_options.Value.SampleFormat != SampleFormat.Float32)
            {
                throw new NotSupportedException("Only SampleFormat.Float32 is supported.");
            }

            _state = AudioServiceState.Initializing;

            nint inputWasapiStreamInfo = nint.Zero;
            nint outputWasapiStreamInfo = nint.Zero;
            try
            {
                ResetCaptureProcessors();
                _captureStreamEpoch = Interlocked.Increment(ref _streamEpochCounter);
                _captureSequence = 0;
                _captureSamplePosition = 0;
                _renderStreamEpoch = checked(_renderStreamEpoch + 1);
                _renderSamplePosition = 0;

                // Wait for any pending reset operation
                if (!_resetEvent.Wait(3000))
                {
                    throw new TimeoutException("Audio queue is being cleaned, cannot start at this moment.");
                }

                CleanupQueues();

                DeviceInfo inputDeviceInfo = GetAudioDevice(ref inputDeviceIndex, AudioDeviceType.Input);
                DeviceInfo outputDeviceInfo = GetAudioDevice(ref outputDeviceIndex, AudioDeviceType.Output);
                _currentInputDeviceIndex = inputDeviceIndex;
                _currentOutputDeviceIndex = outputDeviceIndex;

                if (_options.Value.ChannelCount > inputDeviceInfo.maxInputChannels)
                {
                    throw new ArgumentOutOfRangeException(nameof(_options.Value.ChannelCount),
                        $"Channel count {_options.Value.ChannelCount} exceeds maximum input channels {inputDeviceInfo.maxInputChannels} for device {inputDeviceInfo.name}.");
                }

                double suggestedLatency = _options.Value.SuggestedLatency
                    ?? Math.Max(inputDeviceInfo.defaultLowInputLatency, outputDeviceInfo.defaultLowOutputLatency);

                // WASAPI 设备需要专属 stream info: 共享模式带 AutoConvert
                // (允许 16kHz 等与系统混音格式不一致的流), 独占模式按选项启用
                inputWasapiStreamInfo = Backend.PortAudioBackend.AllocWasapiStreamInfo(inputDeviceIndex, _options.Value.EnableWasapiExclusiveMode);
                outputWasapiStreamInfo = Backend.PortAudioBackend.AllocWasapiStreamInfo(outputDeviceIndex, _options.Value.EnableWasapiExclusiveMode);

                StreamParameters inputParams = new StreamParameters
                {
                    device = inputDeviceIndex,
                    channelCount = (int)_options.Value.ChannelCount,
                    sampleFormat = _options.Value.SampleFormat,
                    suggestedLatency = suggestedLatency,
                    hostApiSpecificStreamInfo = inputWasapiStreamInfo
                };

                StreamParameters outputParams = new StreamParameters
                {
                    device = outputDeviceIndex,
                    channelCount = (int)_options.Value.ChannelCount,
                    sampleFormat = _options.Value.SampleFormat,
                    suggestedLatency = suggestedLatency,
                    hostApiSpecificStreamInfo = outputWasapiStreamInfo
                };

                uint frameSize = CalculateFrameSize(inputDeviceInfo, outputDeviceInfo);

                var processingCts = new CancellationTokenSource();
                _cancellationTokenSource = processingCts;

                var recordChannel = Backend.RealtimeAudioChannel.CreateDropOldest<RecordAudioBuffer>(
                    (int)_options.Value.RecordBufferCapacity,
                    singleReader: true,
                    singleWriter: true,
                    static dropped => dropped.Release());
                _recordChannel = recordChannel;

                var frameChannel = Backend.RealtimeAudioChannel.CreateDropOldest<DuplexAudioFrameBuffer>(
                    (int)_options.Value.DuplexFrameBufferCapacity,
                    singleReader: false,
                    singleWriter: true,
                    static dropped => dropped.Release());
                _duplexFrameChannel = frameChannel;

                // 两个消费循环放专用线程 (LongRunning) 并提优先级 + 注册 MMCSS:
                // TTS/LLM 本地推理满载时, 普通线程池线程会被 ORT 计算线程挤到饥饿,
                // 帧派发一抖, 下游二级 AEC/barge-in 检测的参考对齐立刻劣化。
                // 循环体必须同步阻塞消费 —— async 续体会跳回线程池, 丢掉优先级与 MMCSS 注册。
                var taskStartedSignal = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                _recordProcessingTask = Task.Factory.StartNew(
                    () =>
                    {
                        taskStartedSignal.TrySetResult(true);
                        ProcessRecordQueueLoop(recordChannel, processingCts.Token);
                    },
                    CancellationToken.None,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default);

                _duplexFrameDispatchTask = Task.Factory.StartNew(
                    () => DispatchDuplexFramesLoop(frameChannel, processingCts.Token),
                    CancellationToken.None,
                    TaskCreationOptions.LongRunning,
                    TaskScheduler.Default);

                await taskStartedSignal.Task.WaitAsync(TimeSpan.FromSeconds(1), cancellationToken);

                _stream = new PortAudioStream(
                    inParams: inputParams,
                    outParams: outputParams,
                    sampleRate: _options.Value.SampleRate,
                    framesPerBuffer: frameSize,
                    streamFlags: StreamFlags.ClipOff,
                    callback: OnFullDuplexCallback,
                    userData: nint.Zero
                );

                _stream.Start();

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
                _isPlaying = false;

                CleanupStream();
                Exception? pipelineError = await CleanupRecordPipelineAsync(CancellationToken.None);
                if (pipelineError != null)
                {
                    Debug.WriteLine($"[ERROR] Full-duplex record processing task failed during start cleanup: {pipelineError.Message}\r\n{pipelineError.StackTrace}");
                }

                OnError?.Invoke(this, new AudioErrorEventArgs(ex, "Failed to start full-duplex audio", AudioServiceState.Error));
                throw;
            }
            finally
            {
                // Pa_OpenStream 打开时已把 stream info 拷贝进流自身状态, 此处统一释放
                Backend.PortAudioBackend.FreeWasapiStreamInfo(inputWasapiStreamInfo);
                Backend.PortAudioBackend.FreeWasapiStreamInfo(outputWasapiStreamInfo);
            }
        }

        private uint CalculateFrameSize(DeviceInfo inputDeviceInfo, DeviceInfo outputDeviceInfo)
        {
            if (_options.Value.MaxFrameSize == -1)
            {
                double latency = _options.Value.SuggestedLatency
                    ?? Math.Max(inputDeviceInfo.defaultLowInputLatency, outputDeviceInfo.defaultLowOutputLatency);
                // 先 clamp double 再转 uint: 负 latency (病态设备上报) 直接转 uint 会回绕成巨大值,
                // 使 Math.Max(0, 巨大) 仍为巨大, 导致 Pa_OpenStream 因 framesPerBuffer 过大而失败。
                return (uint)Math.Max(0.0, latency * _options.Value.SampleRate);
            }
            else if (_options.Value.MaxFrameSize == 0)
                return 0;
            else
                return (uint)_options.Value.MaxFrameSize;
        }

        public async Task StopAsync(CancellationToken cancellationToken = default)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                await StopInternalAsync(cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private async Task StopInternalAsync(CancellationToken cancellationToken = default)
        {
            if (_state == AudioServiceState.Idle)
            {
                return;
            }

            if (_state == AudioServiceState.Stopping)
            {
                return;
            }

            _state = AudioServiceState.Stopping;

            try
            {
                _isRecording = false;
                _isPlaying = false;

                CleanupStream();
                Exception? pipelineError = await CleanupRecordPipelineAsync(cancellationToken);

                ResetCaptureProcessors();
                if (pipelineError != null)
                {
                    throw pipelineError;
                }

                _state = AudioServiceState.Idle;
                _currentInputDeviceIndex = -1;
                _currentOutputDeviceIndex = -1;
            }
            catch (Exception ex)
            {
                _state = AudioServiceState.Error;
                OnError?.Invoke(this, new AudioErrorEventArgs(ex, "Failed to stop full-duplex audio", AudioServiceState.Error));
                throw;
            }
        }

        private unsafe StreamCallbackResult OnFullDuplexCallback(nint input, nint output, uint frameCount,
            ref StreamCallbackTimeInfo timeInfo, StreamCallbackFlags statusFlags, nint userData)
        {
            _audioDataRequestedEvent.Reset();
            DuplexAudioFrameBuffer? duplexFrame = null;

            try
            {
                // 检测 xrun
                if ((statusFlags & StreamCallbackFlags.InputOverflow) != 0)
                {
                    Task.Run(() => OnXrun?.Invoke(this, new AudioXrunEventArgs(AudioXrunType.InputOverflow, frameCount)));
                    Debug.WriteLine("[Warning] Input overflow detected (xrun)");
                    ResetRealtimeCaptureProcessorsAfterInputGap();
                }

                bool outputUnderflow =
                    (statusFlags & StreamCallbackFlags.OutputUnderflow) != 0;
                if (outputUnderflow)
                {
                    Task.Run(() => OnXrun?.Invoke(this, new AudioXrunEventArgs(AudioXrunType.OutputUnderflow, frameCount)));
                    Debug.WriteLine("[Warning] Output underflow detected (xrun)");
                }

                // 先填充播放输出，得到与采集时间对齐的远端(render)参考
                ProcessOutput(output, frameCount);

                CaptureRange captureRange = CreateCaptureRange(frameCount);
                RenderSignalTimeline renderTimeline = AdvanceRenderTimeline(
                    ref _renderStreamEpoch,
                    ref _renderSamplePosition,
                    frameCount,
                    _options.Value.ChannelCount,
                    outputUnderflow
                        ? RenderTimelineStatus.Missing
                        : RenderTimelineStatus.Valid,
                    outputUnderflow);
                bool inputDiscontinuity = (statusFlags & StreamCallbackFlags.InputOverflow) != 0;
                duplexFrame = CreateDuplexFrameBufferIfNeeded(
                    input,
                    output,
                    frameCount,
                    in captureRange,
                    in renderTimeline,
                    inputDiscontinuity);

                // 采集链最前端：高通滤波，滤除直流偏置/低频隆隆声后再做后续处理
                IHighPassFilterService? highPassFilter = _enableHpf ? _hpfService : null;
                if (!ApplyHighPassFilter(highPassFilter, input, frameCount))
                    highPassFilter = null;

                CaptureBeforeAec(duplexFrame, input, frameCount);

                // 可选的采集处理（如回声消除）：用 render 参考原地处理麦克风信号
                IAcousticEchoCancellationService? captureProcessor = _enableAec ? _aecService : null;
                bool captureProcessorChanged = _captureTimelineProcessorInitialized &&
                    !ReferenceEquals(_captureTimelineProcessor, captureProcessor);
                if (!_captureTimelineProcessorInitialized || captureProcessorChanged)
                {
                    _captureTimelineMapper.Reset();
                    _captureTimelineProcessor = captureProcessor;
                    _captureTimelineProcessorInitialized = true;
                }
                bool captureProcessed = ApplyCaptureProcessor(
                    captureProcessor,
                    input,
                    output,
                    frameCount,
                    out int captureProcessorDelaySamples);
                int captureSampleCount = checked((int)(frameCount * _options.Value.ChannelCount));
                BufferedCaptureOutput captureOutput = _captureTimelineMapper.Map(
                    in captureRange,
                    captureSampleCount,
                    buffered: captureProcessed && captureProcessorDelaySamples > 0,
                    algorithmicDelaySamples: captureProcessorDelaySamples,
                    hasDiscontinuity: inputDiscontinuity || captureProcessorChanged);

                PublishDuplexFrame(
                    ref duplexFrame,
                    input,
                    frameCount,
                    captureProcessed,
                    in captureOutput);

                // 处理录音输入（AEC 已在回调内完成，DNR 会在录音派发线程上完成）
                ProcessInput(
                    input,
                    frameCount,
                    in captureRange,
                    captureProcessed,
                    captureProcessorDelaySamples,
                    in captureOutput,
                    highPassFilter,
                    captureProcessed ? captureProcessor : null,
                    inputDiscontinuity || captureOutput.MappingReset);

                return StreamCallbackResult.Continue;
            }
            finally
            {
                duplexFrame?.Release();
                _audioDataRequestedEvent.Set();
            }
        }

        /// <summary>
        /// 在回调内调用高通滤波处理器：原地改写 input(麦克风)，滤除直流偏置与低频隆隆声。
        /// 置于采集链最前端（AEC 之前）。在实时线程上执行，吞掉异常以保证音频流不中断。
        /// </summary>
        private unsafe bool ApplyHighPassFilter(
            IHighPassFilterService? proc,
            nint input,
            uint frameCount)
        {
            if (proc == null || frameCount == 0 || input == nint.Zero)
            {
                return false;
            }

            int totalSamples = (int)(frameCount * _options.Value.ChannelCount);
            if (totalSamples == 0)
            {
                return false;
            }

            try
            {
                var capture = new Span<float>(input.ToPointer(), totalSamples);
                proc.Process(capture, _options.Value.ChannelCount, _options.Value.SampleRate);
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] HighPassFilter failed: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// 在回调内调用采集处理器（如 AEC）：以 output 作为对齐的远端参考，原地改写 input(麦克风)。
        /// 在实时线程上执行，吞掉异常以保证音频流不中断。
        /// </summary>
        private unsafe bool ApplyCaptureProcessor(
            IAcousticEchoCancellationService? proc,
            nint input,
            nint output,
            uint frameCount,
            out int algorithmicDelaySamples)
        {
            algorithmicDelaySamples = 0;
            if (proc == null || frameCount == 0 || input == nint.Zero || output == nint.Zero)
            {
                return false;
            }

            int totalSamples = (int)(frameCount * _options.Value.ChannelCount);
            if (totalSamples == 0)
            {
                return false;
            }

            try
            {
                var capture = new Span<float>(input.ToPointer(), totalSamples);
                var render = new ReadOnlySpan<float>(output.ToPointer(), totalSamples);
                proc.Process(capture, render, _options.Value.ChannelCount, _options.Value.SampleRate);
                algorithmicDelaySamples = Math.Max(0, proc.DelaySamples);
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] CaptureProcessor failed: {ex.Message}");
                return false;
            }
        }

        private CaptureRange CreateCaptureRange(uint frameCount)
        {
            long startSample = _captureSamplePosition;
            long sampleCount = checked((long)frameCount * _options.Value.ChannelCount);
            long sequence = ++_captureSequence;
            _captureSamplePosition = checked(startSample + sampleCount);
            return new CaptureRange(
                _captureStreamEpoch,
                sequence,
                sequence,
                startSample,
                _captureSamplePosition,
                Stopwatch.GetTimestamp());
        }

        internal static RenderSignalTimeline AdvanceRenderTimeline(
            ref long streamEpoch,
            ref long samplePosition,
            uint frameCount,
            uint channels,
            RenderTimelineStatus status,
            bool hasDiscontinuity)
        {
            if (streamEpoch <= 0)
                throw new InvalidOperationException("The render stream has not been started.");
            if (channels == 0)
                throw new ArgumentOutOfRangeException(nameof(channels));
            if (status is not RenderTimelineStatus.Valid and
                not RenderTimelineStatus.Gap and
                not RenderTimelineStatus.Missing)
                throw new ArgumentOutOfRangeException(nameof(status));
            if (status == RenderTimelineStatus.Gap && hasDiscontinuity)
            {
                throw new ArgumentException(
                    "A render gap must remain on the current render timeline.",
                    nameof(hasDiscontinuity));
            }

            if (hasDiscontinuity)
            {
                streamEpoch = checked(streamEpoch + 1);
                samplePosition = 0;
            }

            long startSample = samplePosition;
            long sampleCount = checked((long)frameCount * channels);
            samplePosition = checked(startSample + sampleCount);
            return new RenderSignalTimeline(
                streamEpoch,
                startSample,
                samplePosition,
                status,
                hasDiscontinuity);
        }

        private unsafe DuplexAudioFrameBuffer? CreateDuplexFrameBufferIfNeeded(
            nint input,
            nint output,
            uint frameCount,
            in CaptureRange captureRange,
            in RenderSignalTimeline renderTimeline,
            bool hasDiscontinuity)
        {
            if (OnDuplexAudioFrame is null)
                return null;

            var channel = _duplexFrameChannel;
            if (channel is null || input == nint.Zero || output == nint.Zero)
                return null;

            int totalSamples = (int)(frameCount * _options.Value.ChannelCount);
            if (totalSamples <= 0)
                return null;

            DuplexAudioFrameBuffer? buffer = null;
            try
            {
                var render = new ReadOnlySpan<float>(output.ToPointer(), totalSamples);
                var rawCapture = new ReadOnlySpan<float>(input.ToPointer(), totalSamples);
                buffer = DuplexAudioFrameBuffer.Rent(
                    in captureRange,
                    _options.Value.SampleRate,
                    _options.Value.ChannelCount,
                    frameCount,
                    totalSamples);
                buffer.HasDiscontinuity = hasDiscontinuity;
                buffer.RenderTimeline = renderTimeline;
                buffer.PlaybackGain = _currentPlaybackGain;
                render.CopyTo(buffer.RenderReference);
                rawCapture.CopyTo(buffer.CaptureRaw);
                return buffer;
            }
            catch (Exception ex)
            {
                buffer?.Release();
                Debug.WriteLine($"[Warning] Failed to create duplex audio frame snapshot: {ex.Message}");
                return null;
            }
        }

        private unsafe void CaptureBeforeAec(DuplexAudioFrameBuffer? frame, nint input, uint frameCount)
        {
            if (frame is null || input == nint.Zero)
                return;

            int totalSamples = (int)(frameCount * _options.Value.ChannelCount);
            if (totalSamples <= 0)
                return;

            new ReadOnlySpan<float>(input.ToPointer(), totalSamples).CopyTo(frame.CaptureBeforeAec);
        }

        private unsafe void PublishDuplexFrame(
            ref DuplexAudioFrameBuffer? frame,
            nint input,
            uint frameCount,
            bool captureProcessed,
            in BufferedCaptureOutput captureOutput)
        {
            if (frame is null)
                return;

            DuplexAudioFrameBuffer? buffer = frame;
            try
            {
                if (input == nint.Zero)
                    return;

                int totalSamples = (int)(frameCount * _options.Value.ChannelCount);
                if (totalSamples <= 0)
                    return;

                new ReadOnlySpan<float>(input.ToPointer(), totalSamples).CopyTo(buffer.CaptureAfterAec);
                buffer.CaptureProcessed = captureProcessed;
                CaptureRange callbackRange = buffer.CaptureRange;
                CaptureSignalTimeline callbackTimeline =
                    CaptureSignalTimeline.FromWholeBuffer(in callbackRange, totalSamples);
                buffer.Timeline = new DuplexCaptureTimeline(
                    callbackTimeline,
                    callbackTimeline,
                    captureOutput.Timeline);
                buffer.HasDiscontinuity |= captureOutput.MappingReset;

                var channel = _duplexFrameChannel;
                if (channel is null || !channel.Writer.TryWrite(buffer))
                    return;

                frame = null;
                buffer = null;
            }
            finally
            {
                if (buffer != null)
                {
                    buffer.Release();
                    frame = null;
                }
            }
        }

        private unsafe void ProcessInput(
            nint input,
            uint frameCount,
            in CaptureRange captureRange,
            bool captureProcessed,
            int captureProcessorDelaySamples,
            in BufferedCaptureOutput captureOutput,
            IHighPassFilterService? highPassFilter,
            IAcousticEchoCancellationService? captureProcessor,
            bool hasDiscontinuity)
        {
            var recordChannel = _recordChannel;
            var cts = _cancellationTokenSource;

            if (!_isRecording || (OnDataAvailable == null && OnProcessedCaptureFrame == null) || frameCount == 0
                || input == nint.Zero
                || recordChannel == null || cts?.IsCancellationRequested == true)
            {
                return;
            }

            int totalSamples = (int)(frameCount * _options.Value.ChannelCount);
            if (totalSamples == 0)
            {
                return;
            }

            float[] buffer = ArrayPool<float>.Shared.Rent(totalSamples);
            var inputSpan = new ReadOnlySpan<float>(input.ToPointer(), totalSamples);
            inputSpan.CopyTo(buffer);

            var item = new RecordAudioBuffer(
                buffer,
                totalSamples,
                captureRange,
                captureProcessed,
                captureProcessorDelaySamples,
                captureOutput,
                highPassFilter,
                captureProcessor,
                hasDiscontinuity);
            if (!recordChannel.Writer.TryWrite(item))
            {
                item.Release();
                Debug.WriteLine("[Warning] Failed to write audio data to record channel. Channel may be full or closed.");
            }
        }

        private unsafe void ProcessOutput(nint output, uint frameCount)
        {
            int totalSamples = (int)frameCount * (int)_options.Value.ChannelCount;
            float* outputPtr = (float*)output.ToPointer();
            int samplesWritten = 0;

            if (!_resetEvent.IsSet)
            {
                var remainingBuffer = new Span<float>(&outputPtr[samplesWritten], totalSamples - samplesWritten);
                remainingBuffer.Clear();
                return;
            }

            while (samplesWritten < totalSamples && _resetEvent.IsSet)
            {
                if (_currentBuffer != null && _currentBufferPosition >= _currentBuffer.Length)
                {
                    _currentBuffer.Release();
                    _currentBuffer = null;
                    _currentBufferPosition = 0;
                }

                if (_currentBuffer == null)
                {
                    if (_playChannel.Reader.TryRead(out var newBuffer))
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
                            Task.Run(() => OnPlaybackCompleted?.Invoke(this, EventArgs.Empty));
                        }
                        break;
                    }
                }

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

            if (samplesWritten < totalSamples)
            {
                var remainingBuffer = new Span<float>(&outputPtr[samplesWritten], totalSamples - samplesWritten);
                remainingBuffer.Clear();
            }

            ApplyPlaybackGain(outputPtr, totalSamples);
        }

        /// <summary>
        /// 设置播放增益 (0..1)，用于 barge-in 期间对当前 TTS 播放做 ducking（压低而非清空）。1 = 原音量。
        /// 实时回调内按 <see cref="PlaybackGainRampMs"/> 斜坡平滑过渡。线程安全：控制线程调用、音频回调读取。
        /// </summary>
        public void SetPlaybackGain(float gain)
        {
            if (float.IsNaN(gain))
                return;
            Volatile.Write(ref _targetPlaybackGain, Math.Clamp(gain, 0f, 1f));
        }

        /// <summary>
        /// 在回调内对已填充的输出缓冲施加播放增益，按 <see cref="PlaybackGainRampMs"/> 线性斜坡逼近目标增益。
        /// 增益恒为 1 (常态) 时零开销直接返回。施加在 AEC 取 render 参考之前，故远端参考会同步被压低。
        /// </summary>
        private unsafe void ApplyPlaybackGain(float* outputPtr, int totalSamples)
        {
            float target = Volatile.Read(ref _targetPlaybackGain);
            float current = _currentPlaybackGain;
            if (target == 1f && current == 1f)
                return;

            int channels = (int)_options.Value.ChannelCount;
            if (channels < 1)
                channels = 1;
            int rampFrames = Math.Max(1, (int)(_options.Value.SampleRate * PlaybackGainRampMs / 1000f));
            float maxStep = 1f / rampFrames;

            for (int i = 0; i < totalSamples; i += channels)
            {
                if (current != target)
                {
                    float diff = target - current;
                    current += Math.Clamp(diff, -maxStep, maxStep);
                }

                int frameEnd = Math.Min(i + channels, totalSamples);
                for (int c = i; c < frameEnd; c++)
                    outputPtr[c] *= current;
            }

            _currentPlaybackGain = current;
        }

        /// <summary>
        /// 录音消费循环 (DNR + OnDataAvailable 派发)。在专用 LongRunning 线程上同步阻塞消费,
        /// 线程提为 AboveNormal 并注册 MMCSS "Pro Audio", 避免推理满载时喂 ASR 的链路抖动。
        /// </summary>
        private void ProcessRecordQueueLoop(Channel<RecordAudioBuffer> recordChannel, CancellationToken token)
        {
            using var mmcss = Backend.MmcssThreadScope.TryEnter("Pro Audio");
            TrySetCurrentThreadPriority(ThreadPriority.AboveNormal, "duplex record processing");
            var dnrSourceRanges = new CaptureSourceRangeQueue();
            CaptureRange? previousInput = null;
            IDigitalNoiseReductionService? previousDnr = null;
            IHighPassFilterService? previousHighPassFilter = null;
            IAcousticEchoCancellationService? previousCaptureProcessor = null;
            bool forceDiscontinuity = false;
            // The startup reset is best-effort because StartAsync also resets HPF/AEC. Require a
            // successful DNR reset on this single consumer thread before the first stream sample.
            bool dnrResetRequired = true;

            try
            {
                while (WaitToReadBlocking(recordChannel.Reader, token))
                {
                    while (recordChannel.Reader.TryRead(out var item))
                    {
                        try
                        {
                            if (recordChannel.Reader.Count >= 5)
                            {
                                Debug.WriteLine($"[Warning] Audio data queue is over {recordChannel.Reader.Count} items. Please consider improving the processing performance.");
                            }

                            CaptureRange itemRange = item.SourceRange;
                            IDigitalNoiseReductionService? dnr = _enableDnr ? _dnrService : null;
                            bool pipelineChanged = previousInput.HasValue &&
                                (!ReferenceEquals(previousDnr, dnr) ||
                                 !ReferenceEquals(previousHighPassFilter, item.HighPassFilter) ||
                                 !ReferenceEquals(previousCaptureProcessor, item.CaptureProcessor));
                            bool inputGap =
                                item.HasDiscontinuity ||
                                item.CaptureOutput.HasDiscontinuity ||
                                (previousInput.HasValue && !previousInput.Value.IsContiguousWith(in itemRange));
                            if (inputGap || pipelineChanged)
                            {
                                if (previousDnr != null && !ReferenceEquals(previousDnr, dnr))
                                    ResetNoiseReduction(previousDnr);
                                dnrResetRequired = !ResetNoiseReduction(dnr);
                                dnrSourceRanges.Clear();
                                forceDiscontinuity = true;
                            }

                            previousInput = itemRange;
                            previousDnr = dnr;
                            previousHighPassFilter = item.HighPassFilter;
                            previousCaptureProcessor = item.CaptureProcessor;

                            BufferedCaptureOutput captureOutput = item.CaptureOutput;
                            if (!captureOutput.IsValid)
                                continue;

                            int audioOffset = captureOutput.BufferOffset;
                            int captureProcessorOutputLength = captureOutput.SampleCount;
                            CaptureRange captureProcessorOutputRange = captureOutput.SourceRange;

                            int outputLength;
                            CaptureRange outputRange;
                            int algorithmicDelaySamples;
                            if (dnr != null)
                            {
                                if (dnrResetRequired)
                                {
                                    if (!ResetNoiseReduction(dnr))
                                        continue;
                                    dnrResetRequired = false;
                                }
                                dnrSourceRanges.Enqueue(in captureProcessorOutputRange);
                                long maxPendingSamples = checked((long)dnr.DelaySamples +
                                    Math.Max(dnr.FrameSize, captureProcessorOutputLength) * 4L);
                                if (dnrSourceRanges.PendingSamples > maxPendingSamples)
                                {
                                    dnrResetRequired = !ResetNoiseReduction(dnr);
                                    dnrSourceRanges.Clear();
                                    forceDiscontinuity = true;
                                    Debug.WriteLine(
                                        "[Warning] DNR exceeded its declared source backlog; state was reset.");
                                    continue;
                                }
                                if (!TryApplyNoiseReduction(
                                    dnr,
                                    item.Buffer.AsSpan(audioOffset, captureProcessorOutputLength),
                                    out outputLength))
                                {
                                    dnrResetRequired = !ResetNoiseReduction(dnr);
                                    dnrSourceRanges.Clear();
                                    forceDiscontinuity = true;
                                    continue;
                                }

                                if (outputLength <= 0)
                                    continue;

                                if (!dnrSourceRanges.TryDequeue(outputLength, out outputRange))
                                {
                                    dnrResetRequired = !ResetNoiseReduction(dnr);
                                    dnrSourceRanges.Clear();
                                    forceDiscontinuity = true;
                                    Debug.WriteLine("[Warning] DNR output could not be mapped to the capture timeline.");
                                    continue;
                                }

                                algorithmicDelaySamples = checked((int)Math.Min(
                                    int.MaxValue,
                                    Math.Max(0, item.SourceRange.EndSampleExclusive - outputRange.EndSampleExclusive)));
                            }
                            else
                            {
                                dnrSourceRanges.Clear();
                                outputLength = captureProcessorOutputLength;
                                outputRange = captureProcessorOutputRange;
                                algorithmicDelaySamples = checked((int)Math.Min(
                                    int.MaxValue,
                                    Math.Max(0, item.SourceRange.EndSampleExclusive - outputRange.EndSampleExclusive)));
                            }

                            if (outputLength <= 0)
                                continue;

                            var audio = new ReadOnlyMemory<float>(item.Buffer, audioOffset, outputLength);
                            var frame = new ProcessedCaptureFrame(
                                outputRange,
                                _options.Value.SampleRate,
                                _options.Value.ChannelCount,
                                audio,
                                algorithmicDelaySamples,
                                forceDiscontinuity || inputGap || pipelineChanged);
                            forceDiscontinuity = false;

                            ProcessedCaptureFrameHandler? processedHandler = OnProcessedCaptureFrame;
                            if (processedHandler != null)
                                InvokeProcessedCaptureFrameHandlers(processedHandler, in frame);

                            Action<ReadOnlySpan<float>>? legacyHandler = OnDataAvailable;
                            if (legacyHandler != null)
                                InvokeDataAvailableHandlers(legacyHandler, audio.Span);
                        }
                        finally
                        {
                            item.Release();
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine("Audio processing task was canceled.");
            }
        }

        /// <summary>
        /// 双工帧消费循环 (barge-in 检测/二级 AEC 等订阅者)。在专用 LongRunning 线程上同步阻塞
        /// 消费, 线程提为 Highest 并注册 MMCSS "Pro Audio": 该链路对帧到达节奏最敏感,
        /// 推理满载时一旦帧派发迟到, 下游 AEC 参考对齐与近端检测窗口立即劣化。
        /// </summary>
        private void DispatchDuplexFramesLoop(
            Channel<DuplexAudioFrameBuffer> frameChannel,
            CancellationToken token)
        {
            using var mmcss = Backend.MmcssThreadScope.TryEnter("Pro Audio");
            TrySetCurrentThreadPriority(ThreadPriority.Highest, "duplex frame dispatch");
            CaptureRange? previousFrame = null;
            RenderSignalTimeline? previousRender = null;

            try
            {
                while (WaitToReadBlocking(frameChannel.Reader, token))
                {
                    while (frameChannel.Reader.TryRead(out DuplexAudioFrameBuffer? buffer))
                    {
                        try
                        {
                            CaptureRange currentRange = buffer.CaptureRange;
                            if (previousFrame.HasValue && !previousFrame.Value.IsContiguousWith(in currentRange))
                                buffer.HasDiscontinuity = true;
                            previousFrame = currentRange;

                            RenderSignalTimeline currentRender = buffer.RenderTimeline;
                            if (previousRender.HasValue &&
                                !previousRender.Value.IsContiguousWith(in currentRender))
                            {
                                buffer.RenderTimeline = currentRender with
                                {
                                    HasDiscontinuity = true,
                                };
                            }
                            previousRender = buffer.RenderTimeline;

                            DuplexAudioFrameHandler? handler = OnDuplexAudioFrame;
                            if (handler is null)
                                continue;

                            DuplexAudioFrame frame = buffer.ToFrame();
                            InvokeDuplexAudioFrameHandlers(handler, in frame);
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine($"[Warning] Duplex audio frame subscriber failed: {ex.Message}");
                        }
                        finally
                        {
                            buffer.Release();
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                Debug.WriteLine("Duplex audio frame dispatch task was canceled.");
            }
        }

        /// <summary>
        /// 在专用消费线程上同步等待 channel 有数据可读。写端 Complete 且排空后返回 false。
        /// 必须同步阻塞而非 await: async 续体会跳回线程池线程, 丢掉本线程的优先级与 MMCSS 注册。
        /// </summary>
        private static bool WaitToReadBlocking<T>(ChannelReader<T> reader, CancellationToken token)
            => Backend.RealtimeAudioChannel.WaitToReadBlocking(reader, token);

        /// <summary>设置当前线程优先级, 平台不支持时静默降级 (仅 Debug 日志)。</summary>
        private static void TrySetCurrentThreadPriority(ThreadPriority priority, string role)
        {
            try
            {
                Thread.CurrentThread.Priority = priority;
            }
            catch (PlatformNotSupportedException)
            {
                Debug.WriteLine($"[Info] Thread priority not supported on this platform ({role}).");
            }
        }

        private static void InvokeDuplexAudioFrameHandlers(
            DuplexAudioFrameHandler handler,
            in DuplexAudioFrame frame)
        {
            foreach (Delegate subscriber in handler.GetInvocationList())
            {
                try
                {
                    ((DuplexAudioFrameHandler)subscriber).Invoke(in frame);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[Warning] Duplex audio frame subscriber failed: {ex.Message}");
                }
            }
        }

        private static void InvokeProcessedCaptureFrameHandlers(
            ProcessedCaptureFrameHandler handler,
            in ProcessedCaptureFrame frame)
        {
            foreach (Delegate subscriber in handler.GetInvocationList())
            {
                try
                {
                    ((ProcessedCaptureFrameHandler)subscriber).Invoke(in frame);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[Warning] Processed capture subscriber failed: {ex.Message}");
                }
            }
        }

        private static void InvokeDataAvailableHandlers(
            Action<ReadOnlySpan<float>> handler,
            ReadOnlySpan<float> audio)
        {
            foreach (Delegate subscriber in handler.GetInvocationList())
            {
                try
                {
                    ((Action<ReadOnlySpan<float>>)subscriber).Invoke(audio);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[Warning] Audio data subscriber failed: {ex.Message}");
                }
            }
        }

        private async Task<Exception?> CleanupRecordPipelineAsync(CancellationToken cancellationToken)
        {
            var recordChannel = _recordChannel;
            var frameChannel = _duplexFrameChannel;
            var cts = _cancellationTokenSource;
            var processingTask = _recordProcessingTask;
            var frameDispatchTask = _duplexFrameDispatchTask;
            Exception? processingError = null;

            recordChannel?.Writer.TryComplete();
            frameChannel?.Writer.TryComplete();
            bool processingTaskCompleted = processingTask == null;
            bool frameDispatchTaskCompleted = frameDispatchTask == null;

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
                        Debug.WriteLine($"[ERROR] Full-duplex record processing task failed during cleanup: {ex.Message}\r\n{ex.StackTrace}");
                    }
                }
                catch (Exception ex)
                {
                    processingTaskCompleted = true;
                    processingError = ex;
                    Debug.WriteLine($"[ERROR] Full-duplex record processing task failed during cleanup: {ex.Message}\r\n{ex.StackTrace}");
                }
            }

            if (frameDispatchTask != null)
            {
                try
                {
                    await frameDispatchTask.WaitAsync(TimeSpan.FromSeconds(2));
                    frameDispatchTaskCompleted = true;
                }
                catch (OperationCanceledException)
                {
                    frameDispatchTaskCompleted = true;
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
                        await frameDispatchTask.WaitAsync(TimeSpan.FromSeconds(1));
                        frameDispatchTaskCompleted = true;
                    }
                    catch (OperationCanceledException)
                    {
                        frameDispatchTaskCompleted = true;
                    }
                    catch (TimeoutException)
                    {
                    }
                    catch (Exception ex)
                    {
                        frameDispatchTaskCompleted = true;
                        processingError ??= ex;
                        Debug.WriteLine($"[ERROR] Duplex frame dispatch task failed during cleanup: {ex.Message}\r\n{ex.StackTrace}");
                    }
                }
                catch (Exception ex)
                {
                    frameDispatchTaskCompleted = true;
                    processingError ??= ex;
                    Debug.WriteLine($"[ERROR] Duplex frame dispatch task failed during cleanup: {ex.Message}\r\n{ex.StackTrace}");
                }
            }

            if (processingTaskCompleted)
            {
                DrainRecordChannel(recordChannel);
            }
            else
            {
                _ = processingTask!.ContinueWith(
                    _ =>
                    {
                        DrainRecordChannel(recordChannel);
                    },
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
            }

            if (frameDispatchTaskCompleted)
            {
                DrainDuplexFrameChannel(frameChannel);
            }
            else
            {
                DrainDuplexFrameChannel(frameChannel);
                _ = frameDispatchTask!.ContinueWith(
                    _ => DrainDuplexFrameChannel(frameChannel),
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
            }

            if (processingTaskCompleted && frameDispatchTaskCompleted)
                cts?.Dispose();
            else
            {
                _ = Task.WhenAll(
                    processingTask ?? Task.CompletedTask,
                    frameDispatchTask ?? Task.CompletedTask).ContinueWith(
                    _ => cts?.Dispose(),
                    CancellationToken.None,
                    TaskContinuationOptions.ExecuteSynchronously,
                    TaskScheduler.Default);
            }

            _cancellationTokenSource = null;
            _recordProcessingTask = null;
            _duplexFrameDispatchTask = null;
            _recordChannel = null;
            _duplexFrameChannel = null;

            return processingError;
        }

        private static void DrainRecordChannel(Channel<RecordAudioBuffer>? recordChannel)
        {
            if (recordChannel == null)
                return;

            while (recordChannel.Reader.TryRead(out var item))
            {
                item.Release();
            }
        }

        private static void DrainDuplexFrameChannel(Channel<DuplexAudioFrameBuffer>? frameChannel)
        {
            if (frameChannel == null)
                return;

            while (frameChannel.Reader.TryRead(out DuplexAudioFrameBuffer? item))
                item.Release();
        }

        /// <summary>
        /// 在录音派发线程上调用 DNR，流式处理并返回本次实际产出的样本数。
        /// 出错时丢弃可能已被原地改写的当前块，并在下一块以 discontinuity 重建时间轴。
        /// </summary>
        private static bool TryApplyNoiseReduction(
            IDigitalNoiseReductionService proc,
            Span<float> buffer,
            out int produced)
        {
            if (buffer.IsEmpty)
            {
                produced = buffer.Length;
                return true;
            }

            try
            {
                produced = proc.Process(buffer);
                if ((uint)produced > (uint)buffer.Length)
                {
                    throw new InvalidOperationException(
                        $"DNR '{proc.GetType().FullName}' returned {produced} samples for a {buffer.Length}-sample input.");
                }
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] NoiseReduction failed: {ex.Message}");
                produced = 0;
                return false;
            }
        }

        private static bool ResetNoiseReduction(IDigitalNoiseReductionService? service)
        {
            try
            {
                service?.Reset();
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] DNR reset failed: {ex.Message}");
                return false;
            }
        }

        private void ResetCaptureProcessors()
        {
            _captureTimelineMapper.Reset();
            _captureTimelineProcessor = null;
            _captureTimelineProcessorInitialized = false;

            try
            {
                _hpfService?.Reset();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] HPF reset failed: {ex.Message}");
            }

            try
            {
                _aecService?.Reset();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] AEC reset failed: {ex.Message}");
            }

            ResetNoiseReduction(_dnrService);
        }

        private void ResetRealtimeCaptureProcessorsAfterInputGap()
        {
            _captureTimelineMapper.Reset();
            _captureTimelineProcessor = null;
            _captureTimelineProcessorInitialized = false;
            try
            {
                _hpfService?.Reset();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] HPF reset after input overflow failed: {ex.Message}");
            }

            try
            {
                _aecService?.Reset();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[Warning] AEC reset after input overflow failed: {ex.Message}");
            }
        }

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
                if (!_playChannel.Writer.TryWrite(new QueueBuffer(buffer, data.Length)))
                {
                    ArrayPool<float>.Shared.Return(buffer);
                    droppedSamples = data.Length;

                    OnQueueOverflow?.Invoke(this, new AudioQueueOverflowEventArgs(
                        droppedSamples,
                        _playChannel.Reader.Count,
                        (int)_options.Value.PlayMaxQueueSize));

                    Debug.WriteLine($"[Warning] Audio queue overflow: {droppedSamples} samples discarded.");
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

                    CleanupPlayQueue();

                    // 清队列(典型为 barge-in 打断)后恢复满音量，确保下一轮播放不被上一次 ducking 残留压低。
                    // 此处流已停止、回调不在运行，写 _currentPlaybackGain 安全。
                    Volatile.Write(ref _targetPlaybackGain, 1f);
                    _currentPlaybackGain = 1f;

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

            CleanupQueues();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CleanupQueues()
        {
            CleanupPlayQueue();
            // Record channel cleanup is handled by stopping the processing task
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CleanupPlayQueue()
        {
            try
            {
                while (_playChannel.Reader.TryRead(out var buffer))
                {
                    buffer.Release();
                }

                if (_currentBuffer != null)
                {
                    _currentBuffer.Release();
                    _currentBuffer = null;
                    _currentBufferPosition = 0;
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ERROR] Failed to clean up play queue: {ex.Message}\r\n{ex.StackTrace}");
            }
        }

        protected override void OptionsValidator(AudioOptionsBase options)
        {
            base.OptionsValidator(options);

            if (_options.Value.RecordBufferCapacity < 1 || _options.Value.RecordBufferCapacity > short.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.RecordBufferCapacity),
                    $"Record buffer capacity must be between 1-{short.MaxValue}. Current value: {_options.Value.RecordBufferCapacity}");

            if (_options.Value.PlayMaxQueueSize < 10 || _options.Value.PlayMaxQueueSize > 10000)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.PlayMaxQueueSize),
                    $"Play queue size must be between 10-10000. Current value: {_options.Value.PlayMaxQueueSize}");

            if (_options.Value.DuplexFrameBufferCapacity < 1 || _options.Value.DuplexFrameBufferCapacity > 10000)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.DuplexFrameBufferCapacity),
                    $"Duplex frame buffer capacity must be between 1-10000. Current value: {_options.Value.DuplexFrameBufferCapacity}");

            if (_options.Value.MaxFrameSize < -1)
                throw new ArgumentOutOfRangeException(nameof(_options.Value.MaxFrameSize),
                    "MaxFrameSize must be -1 (auto), 0 (PortAudio decides), or a positive integer.");

            if (_options.Value.SuggestedLatency.HasValue &&
                (_options.Value.SuggestedLatency.Value < 0.001 || _options.Value.SuggestedLatency.Value > 1.0))
                throw new ArgumentOutOfRangeException(nameof(_options.Value.SuggestedLatency),
                    "SuggestedLatency must be between 0.001 and 1.0 seconds.");
        }

        // IAudioRecordService 实现
        Task IAudioRecordService.StartRecordingAsync(int deviceIndex, CancellationToken cancellationToken)
            => StartAsync(deviceIndex, _currentOutputDeviceIndex, cancellationToken);

        Task IAudioRecordService.StopRecordingAsync(CancellationToken cancellationToken)
            => StopAsync(cancellationToken);

        Task IAudioRecordService.SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
            => SwitchInputDeviceAsync(deviceIndex, cancellationToken);

        // IAudioPlayService 实现
        Task IAudioPlayService.StartPlayingAsync(int deviceIndex, CancellationToken cancellationToken)
            => StartAsync(_currentInputDeviceIndex, deviceIndex, cancellationToken);

        Task IAudioPlayService.StopPlayingAsync(CancellationToken cancellationToken)
            => StopAsync(cancellationToken);

        Task IAudioPlayService.SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
            => SwitchOutputDeviceAsync(deviceIndex, cancellationToken);

        private async Task SwitchInputDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                if (_state != AudioServiceState.Active)
                {
                    throw new InvalidOperationException("Cannot switch input device when full-duplex audio is not active. Start audio first.");
                }

                if (deviceIndex == _currentInputDeviceIndex)
                {
                    return;
                }

                int outputDeviceIndex = _currentOutputDeviceIndex;
                await StopInternalAsync(cancellationToken);
                await StartInternalAsync(deviceIndex, outputDeviceIndex, cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private async Task SwitchOutputDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
        {
            CheckDisposed();

            await _asyncLock.WaitAsync(cancellationToken);
            try
            {
                if (_state != AudioServiceState.Active)
                {
                    throw new InvalidOperationException("Cannot switch output device when full-duplex audio is not active. Start audio first.");
                }

                if (deviceIndex == _currentOutputDeviceIndex)
                {
                    return;
                }

                int inputDeviceIndex = _currentInputDeviceIndex;
                await StopInternalAsync(cancellationToken);
                await StartInternalAsync(inputDeviceIndex, deviceIndex, cancellationToken);
            }
            finally
            {
                _asyncLock.Release();
            }
        }

        private sealed class RecordAudioBuffer
        {
            private int _released;

            public RecordAudioBuffer(
                float[] buffer,
                int length,
                CaptureRange sourceRange,
                bool captureProcessed,
                int captureProcessorDelaySamples,
                BufferedCaptureOutput captureOutput,
                IHighPassFilterService? highPassFilter,
                IAcousticEchoCancellationService? captureProcessor,
                bool hasDiscontinuity)
            {
                Buffer = buffer;
                Length = length;
                SourceRange = sourceRange;
                CaptureProcessed = captureProcessed;
                CaptureProcessorDelaySamples = captureProcessorDelaySamples;
                CaptureOutput = captureOutput;
                HighPassFilter = highPassFilter;
                CaptureProcessor = captureProcessor;
                HasDiscontinuity = hasDiscontinuity;
            }

            public float[] Buffer { get; }
            public int Length { get; }
            public CaptureRange SourceRange { get; }
            public bool CaptureProcessed { get; }
            public int CaptureProcessorDelaySamples { get; }
            public BufferedCaptureOutput CaptureOutput { get; }
            public IHighPassFilterService? HighPassFilter { get; }
            public IAcousticEchoCancellationService? CaptureProcessor { get; }
            public bool HasDiscontinuity { get; }

            public void Release()
            {
                if (Interlocked.Exchange(ref _released, 1) == 0)
                    ArrayPool<float>.Shared.Return(Buffer, true);
            }
        }

        private sealed class DuplexAudioFrameBuffer
        {
            private int _released;

            private DuplexAudioFrameBuffer(
                CaptureRange captureRange,
                uint sampleRate,
                uint channels,
                uint frameCount,
                int length,
                float[] renderReference,
                float[] captureRaw,
                float[] captureBeforeAec,
                float[] captureAfterAec)
            {
                CaptureRange = captureRange;
                SampleRate = sampleRate;
                Channels = channels;
                FrameCount = frameCount;
                Length = length;
                RenderReferenceArray = renderReference;
                CaptureRawArray = captureRaw;
                CaptureBeforeAecArray = captureBeforeAec;
                CaptureAfterAecArray = captureAfterAec;
            }

            public CaptureRange CaptureRange { get; }
            public long SequenceNumber => CaptureRange.StartSequenceNumber;
            public long TimestampTicks => CaptureRange.TimestampTicks;
            public uint SampleRate { get; }
            public uint Channels { get; }
            public uint FrameCount { get; }
            public int Length { get; }
            public bool CaptureProcessed { get; set; }
            public bool HasDiscontinuity { get; set; }
            public DuplexCaptureTimeline Timeline { get; set; }
            public RenderSignalTimeline RenderTimeline { get; set; }
            public float PlaybackGain { get; set; } = 1f;
            private float[] RenderReferenceArray { get; }
            private float[] CaptureRawArray { get; }
            private float[] CaptureBeforeAecArray { get; }
            private float[] CaptureAfterAecArray { get; }
            public Span<float> RenderReference => RenderReferenceArray.AsSpan(0, Length);
            public Span<float> CaptureRaw => CaptureRawArray.AsSpan(0, Length);
            public Span<float> CaptureBeforeAec => CaptureBeforeAecArray.AsSpan(0, Length);
            public Span<float> CaptureAfterAec => CaptureAfterAecArray.AsSpan(0, Length);

            public static DuplexAudioFrameBuffer Rent(
                in CaptureRange captureRange,
                uint sampleRate,
                uint channels,
                uint frameCount,
                int length)
            {
                float[]? renderReference = null;
                float[]? captureRaw = null;
                float[]? captureBeforeAec = null;
                float[]? captureAfterAec = null;

                try
                {
                    renderReference = ArrayPool<float>.Shared.Rent(length);
                    captureRaw = ArrayPool<float>.Shared.Rent(length);
                    captureBeforeAec = ArrayPool<float>.Shared.Rent(length);
                    captureAfterAec = ArrayPool<float>.Shared.Rent(length);

                    return new DuplexAudioFrameBuffer(
                        captureRange,
                        sampleRate,
                        channels,
                        frameCount,
                        length,
                        renderReference,
                        captureRaw,
                        captureBeforeAec,
                        captureAfterAec);
                }
                catch
                {
                    ReturnIfRented(renderReference);
                    ReturnIfRented(captureRaw);
                    ReturnIfRented(captureBeforeAec);
                    ReturnIfRented(captureAfterAec);
                    throw;
                }
            }

            public DuplexAudioFrame ToFrame()
                => new(
                    SequenceNumber,
                    TimestampTicks,
                    SampleRate,
                    Channels,
                    FrameCount,
                    CaptureProcessed,
                    new ReadOnlyMemory<float>(RenderReferenceArray, 0, Length),
                    new ReadOnlyMemory<float>(CaptureRawArray, 0, Length),
                    new ReadOnlyMemory<float>(CaptureBeforeAecArray, 0, Length),
                    new ReadOnlyMemory<float>(CaptureAfterAecArray, 0, Length),
                    CaptureRange.StreamEpoch,
                    CaptureRange.StartSample,
                    HasDiscontinuity,
                    Timeline,
                    RenderTimeline,
                    PlaybackGain);

            public void Release()
            {
                if (Interlocked.Exchange(ref _released, 1) != 0)
                    return;

                ArrayPool<float>.Shared.Return(RenderReferenceArray, true);
                ArrayPool<float>.Shared.Return(CaptureRawArray, true);
                ArrayPool<float>.Shared.Return(CaptureBeforeAecArray, true);
                ArrayPool<float>.Shared.Return(CaptureAfterAecArray, true);
            }

            private static void ReturnIfRented(float[]? buffer)
            {
                if (buffer != null)
                    ArrayPool<float>.Shared.Return(buffer, true);
            }
        }

        public override void Dispose()
        {
            if (_disposed)
                return;

            _playChannel.Writer.TryComplete();

            try
            {
                StopAsync().GetAwaiter().GetResult();
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
