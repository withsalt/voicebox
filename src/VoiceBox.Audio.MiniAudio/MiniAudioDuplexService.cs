using System.Buffers;
using System.Diagnostics;
using System.Threading.Channels;
using MiniAudioEx.Native;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.MiniAudio.Backend;
using VoiceBox.Audio.MiniAudio.Models;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.Core.Services.HPFService;

namespace VoiceBox.Audio.MiniAudio;

public sealed class MiniAudioDuplexService : MiniAudioServiceBase, IAudioDuplexService
{
    private const float PlaybackGainRampMilliseconds = 15f;

    private readonly IOptions<MiniAudioDuplexOptions> _options;
    private readonly MiniAudioPlaybackQueue _playQueue;
    private readonly BufferedCaptureTimelineMapper _captureTimelineMapper = new();
    private MiniAudioDeviceSession? _session;
    private Channel<CaptureBuffer>? _captureChannel;
    private Channel<DuplexFrameBuffer>? _duplexFrameChannel;
    private Task? _captureTask;
    private Task? _duplexFrameTask;
    private int _currentInputDeviceIndex = -1;
    private int _currentOutputDeviceIndex = -1;
    private long _streamEpoch;
    private long _captureSequence;
    private long _captureSamplePosition;
    private long _renderStreamEpoch;
    private long _renderSamplePosition;
    private int _pendingCaptureDiscontinuity;
    private int _pendingRenderDiscontinuity;
    private float _targetPlaybackGain = 1f;
    private float _currentPlaybackGain = 1f;
    private IAcousticEchoCancellationService? _activeCaptureProcessor;
    private IHighPassFilterService? _activeHighPassFilter;
    private volatile bool _isRecording;
    private volatile bool _enableAec;
    private volatile bool _enableDnr;
    private volatile bool _enableHpf;

    public MiniAudioDuplexService(
        IOptions<MiniAudioDuplexOptions> options,
        IAcousticEchoCancellationService? aecService = null,
        IDigitalNoiseReductionService? dnrService = null,
        IHighPassFilterService? hpfService = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        ValidateBufferOptions(_options.Value);
        ValidateDuplexOptions(_options.Value);

        _aecService = aecService;
        _dnrService = dnrService;
        _hpfService = hpfService;
        _enableAec = _options.Value.EnableAec;
        _enableDnr = _options.Value.EnableDnr;
        _enableHpf = _options.Value.EnableHpf;
        if (_enableAec && _aecService is null)
        {
            throw new InvalidOperationException("AEC is enabled but no IAcousticEchoCancellationService was supplied.");
        }

        if (_enableDnr && _dnrService is null)
        {
            throw new InvalidOperationException("DNR is enabled but no IDigitalNoiseReductionService was supplied.");
        }

        if (_enableHpf && _hpfService is null)
        {
            throw new InvalidOperationException("HPF is enabled but no IHighPassFilterService was supplied.");
        }

        _playQueue = new MiniAudioPlaybackQueue(_options.Value.PlayMaxQueueSize);
    }

    public override AudioOptionsBase Options => _options.Value;

    public bool IsRecording => _isRecording;

    public bool IsPlaying => _playQueue.IsPlaying;

    public int CurrentQueueSize => _playQueue.Count;

    private readonly IAcousticEchoCancellationService? _aecService;

    private readonly IDigitalNoiseReductionService? _dnrService;

    private readonly IHighPassFilterService? _hpfService;

    public bool EnableAec
    {
        get => _enableAec;
        set => _enableAec = value;
    }

    public bool EnableDnr
    {
        get => _enableDnr;
        set => _enableDnr = value;
    }

    public bool EnableHighPassFilter
    {
        get => _enableHpf;
        set => _enableHpf = value;
    }

    public event Action<ReadOnlySpan<float>>? OnDataAvailable;

    public event EventHandler? OnPlaybackCompleted;

    public event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow;

    public event EventHandler<AudioErrorEventArgs>? OnError;

    public event EventHandler<AudioXrunEventArgs>? OnXrun
    {
        // MiniAudio does not expose backend xrun status flags.
        add { }
        remove { }
    }

    public event DuplexAudioFrameHandler? OnDuplexAudioFrame;

    public event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;

    public int GetDefaultInputDevice() => GetDefaultDeviceIndex(AudioDeviceType.Input);

    public int GetDefaultOutputDevice() => GetDefaultDeviceIndex(AudioDeviceType.Output);

    public async Task StartAsync(
        int inputDeviceIndex = -1,
        int outputDeviceIndex = -1,
        CancellationToken cancellationToken = default)
    {
        CheckDisposed();
        await _asyncLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await StartInternalAsync(inputDeviceIndex, outputDeviceIndex, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
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

        bool enqueued = _playQueue.TryEnqueue(data, out droppedSamples);
        if (!enqueued)
        {
            MiniAudioServiceHelpers.InvokeEventHandlers(
                OnQueueOverflow,
                this,
                new AudioQueueOverflowEventArgs(droppedSamples, CurrentQueueSize, (int)_options.Value.PlayMaxQueueSize));
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
            _playQueue.Clear();
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

    Task IAudioRecordService.StartRecordingAsync(int deviceIndex, CancellationToken cancellationToken)
        => StartAsync(deviceIndex, _currentOutputDeviceIndex, cancellationToken);

    Task IAudioRecordService.StopRecordingAsync(CancellationToken cancellationToken)
        => StopAsync(cancellationToken);

    Task IAudioRecordService.SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
        => SwitchInputDeviceAsync(deviceIndex, cancellationToken);

    Task IAudioPlayService.StartPlayingAsync(int deviceIndex, CancellationToken cancellationToken)
        => StartAsync(_currentInputDeviceIndex, deviceIndex, cancellationToken);

    Task IAudioPlayService.StopPlayingAsync(CancellationToken cancellationToken)
        => StopAsync(cancellationToken);

    Task IAudioPlayService.SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
        => SwitchOutputDeviceAsync(deviceIndex, cancellationToken);

    public override void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            StopAsync().GetAwaiter().GetResult();
        }
        catch
        {
        }

        _playQueue.Dispose();
        base.Dispose();
    }

    private async Task StartInternalAsync(int inputDeviceIndex, int outputDeviceIndex, CancellationToken cancellationToken)
    {
        if (_state == AudioServiceState.Active)
        {
            return;
        }

        if (_state is AudioServiceState.Initializing or AudioServiceState.Stopping)
        {
            throw new InvalidOperationException($"Cannot start full-duplex audio while in {_state} state.");
        }

        if (_options.Value.SampleFormat != SampleFormat.Float32)
        {
            throw new NotSupportedException("Only SampleFormat.Float32 is supported.");
        }

        _state = AudioServiceState.Initializing;
        try
        {
            MiniAudioDeviceSelection input = ResolveDevice(inputDeviceIndex, AudioDeviceType.Input);
            MiniAudioDeviceSelection output = ResolveDevice(outputDeviceIndex, AudioDeviceType.Output, input.Backend);
            if (input.Backend != output.Backend)
            {
                throw new InvalidOperationException("Full-duplex input and output devices must use the same MiniAudio backend.");
            }

            ResetCaptureProcessors();
            _captureSamplePosition = 0;
            _captureSequence = 0;
            _renderSamplePosition = 0;
            _streamEpoch = Interlocked.Increment(ref _streamEpoch);
            _renderStreamEpoch = Interlocked.Increment(ref _renderStreamEpoch);
            _playQueue.Clear();
            Interlocked.Exchange(ref _pendingCaptureDiscontinuity, 0);
            Interlocked.Exchange(ref _pendingRenderDiscontinuity, 0);

            Channel<CaptureBuffer> captureChannel = CreateCaptureChannel(_options.Value.RecordBufferCapacity);
            Channel<DuplexFrameBuffer> frameChannel = CreateFrameChannel(_options.Value.DuplexFrameBufferCapacity);
            _captureChannel = captureChannel;
            _duplexFrameChannel = frameChannel;

            // Both consumers run on dedicated threads with raised priority and an MMCSS
            // registration: when local TTS/LLM inference saturates the cores, an ordinary thread
            // pool thread gets starved, and one late frame immediately degrades the downstream
            // AEC reference alignment and barge-in detection. The loop bodies must consume
            // synchronously — an async continuation hops back to the thread pool and loses both
            // the priority and the MMCSS registration.
            var taskStartedSignal = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            _captureTask = Task.Factory.StartNew(
                () =>
                {
                    taskStartedSignal.TrySetResult(true);
                    ProcessCaptureQueueLoop(captureChannel);
                },
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
            _duplexFrameTask = Task.Factory.StartNew(
                () => DispatchDuplexFramesLoop(frameChannel),
                CancellationToken.None,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default);
            await taskStartedSignal.Task.WaitAsync(TimeSpan.FromSeconds(1), cancellationToken).ConfigureAwait(false);

            _session = new MiniAudioDeviceSession(
                output,
                input,
                _options.Value.SampleRate,
                _options.Value.ChannelCount,
                _options.Value.MaxFrameSize,
                _options.Value.SuggestedLatency,
                _options.Value.EnableWasapiExclusiveMode,
                DeviceCallback);
            _currentInputDeviceIndex = input.AudioDevice.Id;
            _currentOutputDeviceIndex = output.AudioDevice.Id;
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
            _duplexFrameChannel?.Writer.TryComplete();
            await DrainPipelineAsync().ConfigureAwait(false);
            _captureChannel = null;
            _duplexFrameChannel = null;
            _currentInputDeviceIndex = -1;
            _currentOutputDeviceIndex = -1;
            _state = AudioServiceState.Error;
            RaiseError(exception, "Failed to start full-duplex audio.");
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
        _duplexFrameChannel?.Writer.TryComplete();
        await DrainPipelineAsync().ConfigureAwait(false);
        _captureChannel = null;
        _duplexFrameChannel = null;
        _playQueue.Clear();
        _currentInputDeviceIndex = -1;
        _currentOutputDeviceIndex = -1;
        if (stopError is not null)
        {
            _state = AudioServiceState.Error;
            RaiseError(stopError, "Failed to stop full-duplex audio.");
            throw stopError;
        }

        _state = AudioServiceState.Idle;
    }

    private unsafe void DeviceCallback(ma_device_ptr device, IntPtr output, IntPtr input, uint frameCount)
    {
        if (frameCount == 0)
        {
            return;
        }

        int sampleCount = checked((int)(frameCount * _options.Value.ChannelCount));
        if (output == IntPtr.Zero || input == IntPtr.Zero)
        {
            if (output != IntPtr.Zero)
            {
                new Span<float>((void*)output, sampleCount).Clear();
            }

            Interlocked.Exchange(ref _pendingCaptureDiscontinuity, 1);
            Interlocked.Exchange(ref _pendingRenderDiscontinuity, 1);
            return;
        }

        float[]? captureBuffer = null;
        DuplexFrameBuffer? frameBuffer = null;
        try
        {
            Span<float> outputSamples = new((void*)output, sampleCount);
            bool playbackCompleted = _playQueue.Fill(outputSamples);
            ApplyPlaybackGain(outputSamples);
            if (playbackCompleted)
            {
                _ = Task.Run(() => MiniAudioServiceHelpers.InvokeEventHandlers(OnPlaybackCompleted, this));
            }

            CaptureRange captureRange = CreateCaptureRange(frameCount);
            bool renderDiscontinuity = Interlocked.Exchange(ref _pendingRenderDiscontinuity, 0) != 0;
            RenderSignalTimeline renderTimeline = AdvanceRenderTimeline(frameCount, renderDiscontinuity);
            bool hasDiscontinuity = Interlocked.Exchange(ref _pendingCaptureDiscontinuity, 0) != 0;
            ReadOnlySpan<float> rawCapture = new((void*)input, sampleCount);
            bool needsRecord = OnDataAvailable is not null || OnProcessedCaptureFrame is not null;
            bool needsDuplexFrame = OnDuplexAudioFrame is not null;
            if (!needsRecord && !needsDuplexFrame)
            {
                // Processing and its source mapper skip this callback together.
                // Resume with a new mapping instead of joining noncontiguous ranges.
                Interlocked.Exchange(ref _pendingCaptureDiscontinuity, 1);
                return;
            }

            if (needsDuplexFrame)
            {
                frameBuffer = DuplexFrameBuffer.Rent(sampleCount);
                outputSamples.CopyTo(frameBuffer.RenderReference);
                rawCapture.CopyTo(frameBuffer.CaptureRaw);
                frameBuffer.CaptureRange = captureRange;
                frameBuffer.SampleRate = _options.Value.SampleRate;
                frameBuffer.Channels = _options.Value.ChannelCount;
                frameBuffer.FrameCount = frameCount;
                frameBuffer.RenderTimeline = renderTimeline;
                frameBuffer.PlaybackGain = _currentPlaybackGain;
                frameBuffer.HasDiscontinuity = hasDiscontinuity;
            }

            // The snapshot owns a separate after-AEC region. When no record
            // consumer exists, it can also hold the in-place processing buffer.
            Span<float> captureSamples;
            if (needsRecord)
            {
                // Round up to power of 2 for better ArrayPool bucketing
                int rentSize = sampleCount <= 16 ? sampleCount : RoundUpToPowerOf2(sampleCount);
                captureBuffer = ArrayPool<float>.Shared.Rent(rentSize);
                captureSamples = captureBuffer.AsSpan(0, sampleCount);
            }
            else
            {
                captureSamples = frameBuffer!.CaptureAfterAec;
            }

            rawCapture.CopyTo(captureSamples);

            IHighPassFilterService? highPassFilter = _enableHpf ? _hpfService : null;
            IAcousticEchoCancellationService? captureProcessor = _enableAec ? _aecService : null;
            if (!ReferenceEquals(_activeCaptureProcessor, captureProcessor) || !ReferenceEquals(_activeHighPassFilter, highPassFilter))
            {
                _captureTimelineMapper.Reset();
                SafeReset(_activeCaptureProcessor);
                SafeReset(_activeHighPassFilter);
                if (!ReferenceEquals(_activeCaptureProcessor, captureProcessor))
                {
                    SafeReset(captureProcessor);
                }

                if (!ReferenceEquals(_activeHighPassFilter, highPassFilter))
                {
                    SafeReset(highPassFilter);
                }

                _activeCaptureProcessor = captureProcessor;
                _activeHighPassFilter = highPassFilter;
                hasDiscontinuity = true;
            }

            if (highPassFilter is not null)
            {
                try
                {
                    highPassFilter.Process(captureSamples, _options.Value.ChannelCount, _options.Value.SampleRate);
                }
                catch (Exception exception)
                {
                    Debug.WriteLine($"MiniAudio HPF failed: {exception.Message}");
                    highPassFilter = null;
                }
            }

            if (frameBuffer is not null)
            {
                captureSamples.CopyTo(frameBuffer.CaptureBeforeAec);
            }

            bool captureProcessed = false;
            int processorDelaySamples = 0;
            if (captureProcessor is not null)
            {
                try
                {
                    captureProcessor.Process(
                        captureSamples,
                        outputSamples,
                        _options.Value.ChannelCount,
                        _options.Value.SampleRate);
                    captureProcessed = true;
                    processorDelaySamples = Math.Max(0, captureProcessor.DelaySamples);
                }
                catch (Exception exception)
                {
                    Debug.WriteLine($"MiniAudio AEC failed: {exception.Message}");
                    captureProcessor = null;
                }
            }

            BufferedCaptureOutput captureOutput = _captureTimelineMapper.Map(
                in captureRange,
                sampleCount,
                buffered: captureProcessed && processorDelaySamples > 0,
                algorithmicDelaySamples: processorDelaySamples,
                hasDiscontinuity: hasDiscontinuity);

            if (frameBuffer is not null)
            {
                if (needsRecord)
                {
                    captureSamples.CopyTo(frameBuffer.CaptureAfterAec);
                }
                frameBuffer.CaptureProcessed = captureProcessed;
                frameBuffer.Timeline = new DuplexCaptureTimeline(
                    CaptureSignalTimeline.FromWholeBuffer(in captureRange, sampleCount),
                    CaptureSignalTimeline.FromWholeBuffer(in captureRange, sampleCount),
                    captureOutput.Timeline);
                frameBuffer.HasDiscontinuity |= captureOutput.MappingReset;
                Channel<DuplexFrameBuffer>? frameChannel = _duplexFrameChannel;
                if (frameChannel is not null && frameChannel.Writer.TryWrite(frameBuffer))
                {
                    frameBuffer = null;
                }
            }

            if (needsRecord && captureOutput.IsValid)
            {
                var captureItem = new CaptureBuffer(
                    captureBuffer!,
                    captureRange,
                    captureOutput,
                    hasDiscontinuity || captureOutput.MappingReset);
                captureBuffer = null;
                Channel<CaptureBuffer>? captureChannel = _captureChannel;
                if (captureChannel is null || !captureChannel.Writer.TryWrite(captureItem))
                {
                    captureItem.Release();
                    Interlocked.Exchange(ref _pendingCaptureDiscontinuity, 1);
                }
            }
        }
        catch (Exception exception)
        {
            RaiseErrorAsync(exception, "Full-duplex callback failed.");
            Interlocked.Exchange(ref _pendingCaptureDiscontinuity, 1);
        }
        finally
        {
            if (captureBuffer is not null)
            {
                ArrayPool<float>.Shared.Return(captureBuffer, clearArray: true);
            }

            frameBuffer?.Release();
        }
    }

    // Overflow keeps the newest audio: the oldest queued block is dropped and returned to the
    // pool. The resulting omission is reported downstream by the source-range contiguity checks
    // (both loops below), so a slow consumer can never deliver stale audio while newer audio is
    // discarded.
    private static Channel<CaptureBuffer> CreateCaptureChannel(uint capacity)
        => Channel.CreateBounded<CaptureBuffer>(
            new BoundedChannelOptions((int)capacity)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true,
                // WaitToReadBlocking consumes the ValueTask through AsTask().GetResult().
                // Allow the channel to complete that wait synchronously so a saturated thread
                // pool cannot delay the dedicated audio consumer.
                AllowSynchronousContinuations = true,
            },
            static dropped => dropped.Release());

    private static Channel<DuplexFrameBuffer> CreateFrameChannel(uint capacity)
        => Channel.CreateBounded<DuplexFrameBuffer>(
            new BoundedChannelOptions((int)capacity)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = true,
                AllowSynchronousContinuations = true,
            },
            static dropped => dropped.Release());

    // Capture consumer (DNR + processed-frame/legacy dispatch). Runs on a dedicated thread that
    // consumes synchronously so the MMCSS registration and raised priority keep applying.
    private void ProcessCaptureQueueLoop(Channel<CaptureBuffer> channel)
    {
        using var mmcss = MmcssThreadScope.TryEnter("Pro Audio");
        TrySetCurrentThreadPriority(ThreadPriority.AboveNormal, "duplex record processing");
        var sourceRanges = new CaptureSourceRangeQueue();
        IDigitalNoiseReductionService? activeDnr = null;
        CaptureRange? previousRange = null;
        bool forceDiscontinuity = false;
        try
        {
            while (WaitToReadBlocking(channel.Reader))
            {
                while (channel.Reader.TryRead(out CaptureBuffer item))
                {
                    try
                    {
                        CaptureRange currentRange = item.CaptureRange;
                        bool inputGap = item.HasDiscontinuity ||
                            item.CaptureOutput.HasDiscontinuity ||
                            (previousRange.HasValue && !previousRange.Value.IsContiguousWith(in currentRange));
                        previousRange = currentRange;
                        IDigitalNoiseReductionService? dnr = _enableDnr ? _dnrService : null;
                        bool pipelineChanged = !ReferenceEquals(activeDnr, dnr);
                        if (inputGap || pipelineChanged)
                        {
                            SafeReset(activeDnr);
                            SafeReset(dnr);
                            sourceRanges.Clear();
                            forceDiscontinuity = true;
                        }

                        activeDnr = dnr;
                        CaptureSignalTimeline sourceTimeline = item.CaptureOutput.Timeline;
                        if (!sourceTimeline.IsValid)
                        {
                            continue;
                        }

                        int inputOffset = sourceTimeline.BufferOffset;
                        int inputLength = sourceTimeline.SampleCount;
                        int outputLength = inputLength;
                        CaptureRange outputRange = sourceTimeline.SourceRange;
                        int delaySamples = 0;
                        if (dnr is not null)
                        {
                            sourceRanges.Enqueue(in outputRange);
                            Span<float> dnrInput = item.Buffer.AsSpan(inputOffset, inputLength);
                            outputLength = dnr.Process(dnrInput);
                            if ((uint)outputLength > (uint)inputLength)
                            {
                                throw new InvalidOperationException("DNR returned more samples than it consumed.");
                            }

                            if (outputLength <= 0 || !sourceRanges.TryDequeue(outputLength, out outputRange))
                            {
                                continue;
                            }

                            delaySamples = checked((int)Math.Min(
                                int.MaxValue,
                                Math.Max(0, item.CaptureRange.EndSampleExclusive - outputRange.EndSampleExclusive)));
                        }

                        var outputAudio = new ReadOnlyMemory<float>(item.Buffer, inputOffset, outputLength);
                        var processedFrame = new ProcessedCaptureFrame(
                            outputRange,
                            _options.Value.SampleRate,
                            _options.Value.ChannelCount,
                            outputAudio,
                            delaySamples,
                            forceDiscontinuity || inputGap || pipelineChanged);
                        forceDiscontinuity = false;
                        MiniAudioServiceHelpers.InvokeProcessedFrameHandlers(OnProcessedCaptureFrame, in processedFrame);
                        MiniAudioServiceHelpers.InvokeDataHandlers(OnDataAvailable, outputAudio.Span);
                    }
                    catch (Exception exception)
                    {
                        SafeReset(activeDnr);
                        sourceRanges.Clear();
                        activeDnr = null;
                        forceDiscontinuity = true;
                        RaiseError(exception, "Capture processing failed.");
                    }
                    finally
                    {
                        item.Release();
                    }
                }
            }
        }
        catch (Exception exception)
        {
            RaiseError(exception, "Capture processing queue failed.");
        }
    }

    // Duplex frame consumer (barge-in detection, secondary AEC, ...). The frame path is the most
    // scheduling sensitive one, so it runs at the highest priority on its own MMCSS registered
    // thread.
    private void DispatchDuplexFramesLoop(Channel<DuplexFrameBuffer> channel)
    {
        using var mmcss = MmcssThreadScope.TryEnter("Pro Audio");
        TrySetCurrentThreadPriority(ThreadPriority.Highest, "duplex frame dispatch");
        CaptureRange? previousCapture = null;
        RenderSignalTimeline? previousRender = null;
        try
        {
            while (WaitToReadBlocking(channel.Reader))
            {
                while (channel.Reader.TryRead(out DuplexFrameBuffer? buffer))
                {
                    try
                    {
                        CaptureRange currentCapture = buffer.CaptureRange;
                        RenderSignalTimeline currentRender = buffer.RenderTimeline;
                        if (previousCapture.HasValue && !previousCapture.Value.IsContiguousWith(in currentCapture))
                        {
                            buffer.HasDiscontinuity = true;
                        }

                        if (previousRender.HasValue && !previousRender.Value.IsContiguousWith(in currentRender))
                        {
                            buffer.HasDiscontinuity = true;
                        }

                        previousCapture = currentCapture;
                        previousRender = currentRender;
                        DuplexAudioFrame frame = buffer.ToFrame();
                        MiniAudioServiceHelpers.InvokeDuplexFrameHandlers(OnDuplexAudioFrame, in frame);
                    }
                    finally
                    {
                        buffer.Release();
                    }
                }
            }
        }
        catch (Exception exception)
        {
            RaiseError(exception, "Duplex frame queue failed.");
        }
    }

    // Blocks the consumer thread until the reader has data. The dedicated threads must not use an
    // async continuation: it would hop back to the thread pool and lose the thread's priority and
    // MMCSS registration.
    private static bool WaitToReadBlocking<T>(ChannelReader<T> reader)
    {
        ValueTask<bool> wait = reader.WaitToReadAsync();
        return wait.IsCompleted ? wait.Result : wait.AsTask().GetAwaiter().GetResult();
    }

    private static void TrySetCurrentThreadPriority(ThreadPriority priority, string role)
    {
        try
        {
            Thread.CurrentThread.Priority = priority;
        }
        catch (PlatformNotSupportedException)
        {
            Debug.WriteLine($"MiniAudio thread priority is not supported on this platform ({role}).");
        }
    }

    private async Task DrainPipelineAsync()
    {
        Task? captureTask = _captureTask;
        Task? frameTask = _duplexFrameTask;
        _captureTask = null;
        _duplexFrameTask = null;
        if (captureTask is not null)
        {
            await captureTask.ConfigureAwait(false);
        }

        if (frameTask is not null)
        {
            await frameTask.ConfigureAwait(false);
        }

        while (_captureChannel?.Reader.TryRead(out CaptureBuffer capture) == true)
        {
            capture.Release();
        }

        while (_duplexFrameChannel?.Reader.TryRead(out DuplexFrameBuffer? frame) == true)
        {
            frame.Release();
        }
    }

    private CaptureRange CreateCaptureRange(uint frameCount)
    {
        long sequence = checked(++_captureSequence);
        long startSample = _captureSamplePosition;
        long endSample = checked(startSample + (long)frameCount * _options.Value.ChannelCount);
        _captureSamplePosition = endSample;
        return new CaptureRange(_streamEpoch, sequence, sequence, startSample, endSample, Stopwatch.GetTimestamp());
    }

    private RenderSignalTimeline AdvanceRenderTimeline(uint frameCount, bool hasDiscontinuity)
    {
        if (hasDiscontinuity)
        {
            _renderStreamEpoch = checked(_renderStreamEpoch + 1);
            _renderSamplePosition = 0;
        }

        long startSample = _renderSamplePosition;
        long endSample = checked(startSample + (long)frameCount * _options.Value.ChannelCount);
        _renderSamplePosition = endSample;
        return new RenderSignalTimeline(
            _renderStreamEpoch,
            startSample,
            endSample,
            RenderTimelineStatus.Valid,
            hasDiscontinuity);
    }

    private void ApplyPlaybackGain(Span<float> samples)
    {
        float target = Volatile.Read(ref _targetPlaybackGain);
        float current = _currentPlaybackGain;

        // Fast path: already at target
        if (MathF.Abs(target - current) < 0.0001f)
        {
            _currentPlaybackGain = target;
            if (target != 1f)
            {
                MiniAudioServiceHelpers.ScaleSamples(samples, target);
            }
            return;
        }

        // Ramp path: precompute step
        float rampSamples = _options.Value.SampleRate * _options.Value.ChannelCount * PlaybackGainRampMilliseconds / 1000f;
        float gainStep = (target - current) / Math.Max(1f, rampSamples);

        _currentPlaybackGain = MiniAudioServiceHelpers.ApplyGainRamp(samples, current, target, gainStep);
    }

    private void ResetCaptureProcessors()
    {
        _captureTimelineMapper.Reset();
        _activeCaptureProcessor = null;
        _activeHighPassFilter = null;
        SafeReset(_aecService);
        SafeReset(_hpfService);
        SafeReset(_dnrService);
    }

    private static void SafeReset(IAcousticEchoCancellationService? service)
    {
        try
        {
            service?.Reset();
        }
        catch (Exception exception)
        {
            Debug.WriteLine($"MiniAudio AEC reset failed: {exception.Message}");
        }
    }

    private static void SafeReset(IHighPassFilterService? service)
    {
        try
        {
            service?.Reset();
        }
        catch (Exception exception)
        {
            Debug.WriteLine($"MiniAudio HPF reset failed: {exception.Message}");
        }
    }

    private static void SafeReset(IDigitalNoiseReductionService? service)
    {
        try
        {
            service?.Reset();
        }
        catch (Exception exception)
        {
            Debug.WriteLine($"MiniAudio DNR reset failed: {exception.Message}");
        }
    }

    private async Task SwitchInputDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        CheckDisposed();
        await _asyncLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_state != AudioServiceState.Active)
            {
                throw new InvalidOperationException("Cannot switch input device while full-duplex audio is stopped.");
            }

            int currentOutput = _currentOutputDeviceIndex;
            if (deviceIndex != _currentInputDeviceIndex)
            {
                await StopInternalAsync().ConfigureAwait(false);
                await StartInternalAsync(deviceIndex, currentOutput, cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    private async Task SwitchOutputDeviceAsync(int deviceIndex, CancellationToken cancellationToken)
    {
        CheckDisposed();
        await _asyncLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_state != AudioServiceState.Active)
            {
                throw new InvalidOperationException("Cannot switch output device while full-duplex audio is stopped.");
            }

            int currentInput = _currentInputDeviceIndex;
            if (deviceIndex != _currentOutputDeviceIndex)
            {
                await StopInternalAsync().ConfigureAwait(false);
                await StartInternalAsync(currentInput, deviceIndex, cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            _asyncLock.Release();
        }
    }

    private static void ValidateDuplexOptions(MiniAudioDuplexOptions options)
    {
        if (options.RecordBufferCapacity is < 1 or > 32767u)
        {
            throw new ArgumentOutOfRangeException(nameof(options.RecordBufferCapacity));
        }

        if (options.PlayMaxQueueSize is < 10 or > 10000)
        {
            throw new ArgumentOutOfRangeException(nameof(options.PlayMaxQueueSize));
        }

        if (options.DuplexFrameBufferCapacity is < 1 or > 10000)
        {
            throw new ArgumentOutOfRangeException(nameof(options.DuplexFrameBufferCapacity));
        }
    }

    private void RaiseError(Exception exception, string message)
        => MiniAudioServiceHelpers.InvokeEventHandlers(
            OnError,
            this,
            new AudioErrorEventArgs(exception, message, _state));

    private void RaiseErrorAsync(Exception exception, string message)
        => _ = Task.Run(() => RaiseError(exception, message));

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

    // A successful TryWrite transfers sole ownership to the processing queue.
    private readonly struct CaptureBuffer(
        float[] buffer,
        CaptureRange captureRange,
        BufferedCaptureOutput captureOutput,
        bool hasDiscontinuity)
    {
        internal float[] Buffer { get; } = buffer;
        internal CaptureRange CaptureRange { get; } = captureRange;
        internal BufferedCaptureOutput CaptureOutput { get; } = captureOutput;
        internal bool HasDiscontinuity { get; } = hasDiscontinuity;

        internal void Release()
        {
            ArrayPool<float>.Shared.Return(Buffer, clearArray: true);
        }
    }

    private sealed class DuplexFrameBuffer
    {
        // Four pooled regions up to this size fit in a 16,384-float bucket,
        // below the LOH threshold. Larger frames retain separate arrays so
        // combining the snapshots does not move existing SOH arrays to the LOH.
        private const int MaxContiguousSamples = 4096;
        private int _released;

        private DuplexFrameBuffer(int length, float[] samples, float[]? raw = null, float[]? before = null, float[]? after = null)
        {
            Length = length;
            Samples = samples;
            RawArray = raw;
            BeforeAecArray = before;
            AfterAecArray = after;
        }

        internal int Length { get; }
        internal CaptureRange CaptureRange { get; set; }
        internal uint SampleRate { get; set; }
        internal uint Channels { get; set; }
        internal uint FrameCount { get; set; }
        internal bool CaptureProcessed { get; set; }
        internal bool HasDiscontinuity { get; set; }
        internal DuplexCaptureTimeline Timeline { get; set; }
        internal RenderSignalTimeline RenderTimeline { get; set; }
        internal float PlaybackGain { get; set; }
        // Small frames use four disjoint regions in one array. Large frames
        // retain separate arrays. Only the logical length is ever published.
        private float[] Samples { get; }
        private float[]? RawArray { get; }
        private float[]? BeforeAecArray { get; }
        private float[]? AfterAecArray { get; }
        internal Span<float> RenderReference => Samples.AsSpan(0, Length);
        internal Span<float> CaptureRaw => (RawArray ?? Samples).AsSpan(RawArray is null ? Length : 0, Length);
        internal Span<float> CaptureBeforeAec => (BeforeAecArray ?? Samples).AsSpan(BeforeAecArray is null ? 2 * Length : 0, Length);
        internal Span<float> CaptureAfterAec => (AfterAecArray ?? Samples).AsSpan(AfterAecArray is null ? 3 * Length : 0, Length);

        internal static DuplexFrameBuffer Rent(int length)
        {
            float[] samples = ArrayPool<float>.Shared.Rent(length <= MaxContiguousSamples ? checked(4 * length) : length);
            float[]? raw = null;
            float[]? before = null;
            float[]? after = null;
            try
            {
                if (length > MaxContiguousSamples)
                {
                    raw = ArrayPool<float>.Shared.Rent(length);
                    before = ArrayPool<float>.Shared.Rent(length);
                    after = ArrayPool<float>.Shared.Rent(length);
                }

                return new DuplexFrameBuffer(length, samples, raw, before, after);
            }
            catch
            {
                ArrayPool<float>.Shared.Return(samples, clearArray: true);
                Return(raw);
                Return(before);
                Return(after);
                throw;
            }
        }

        internal DuplexAudioFrame ToFrame()
            => new(
                CaptureRange.StartSequenceNumber,
                CaptureRange.TimestampTicks,
                SampleRate,
                Channels,
                FrameCount,
                CaptureProcessed,
                new ReadOnlyMemory<float>(Samples, 0, Length),
                new ReadOnlyMemory<float>(RawArray ?? Samples, RawArray is null ? Length : 0, Length),
                new ReadOnlyMemory<float>(BeforeAecArray ?? Samples, BeforeAecArray is null ? 2 * Length : 0, Length),
                new ReadOnlyMemory<float>(AfterAecArray ?? Samples, AfterAecArray is null ? 3 * Length : 0, Length),
                CaptureRange.StreamEpoch,
                CaptureRange.StartSample,
                HasDiscontinuity,
                Timeline,
                RenderTimeline,
                PlaybackGain);

        internal void Release()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
            {
                return;
            }

            ArrayPool<float>.Shared.Return(Samples, clearArray: true);
            Return(RawArray);
            Return(BeforeAecArray);
            Return(AfterAecArray);
        }

        private static void Return(float[]? buffer)
        {
            if (buffer is not null)
            {
                ArrayPool<float>.Shared.Return(buffer, clearArray: true);
            }
        }
    }
}
