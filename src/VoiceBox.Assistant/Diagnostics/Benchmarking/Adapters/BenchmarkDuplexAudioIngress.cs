using VoiceBox.Assistant.Diagnostics.Benchmarking.Contracts;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Diagnostics.Benchmarking.Adapters
{
    /// <summary>
    /// Benchmark-only replacement for a hardware duplex service. It accepts dataset microphone
    /// and render frames, runs the configured production capture processor, then publishes the
    /// same frame contracts consumed by <see cref="Runtime.AssistantRuntime"/>.
    /// </summary>
    internal sealed class BenchmarkDuplexAudioIngress : IAudioDuplexService, ICapturePositionView
    {
        private readonly IAcousticEchoCancellationService _aecService;
        private readonly TimeProvider _timeProvider;
        private readonly BufferedCaptureTimelineMapper _captureTimelineMapper = new();
        private readonly object _playbackCaptureSync = new();
        private readonly List<float> _capturedPlayback = [];
        private readonly List<float> _playbackQueue = [];

        /// <summary>
        /// Samples committed to the device, in emission order and with playback gain already
        /// applied. Acts as the committed-output delay line: the harness reads from
        /// <see cref="_emitCursor"/> while new samples are appended at the tail.
        /// </summary>
        private readonly List<float> _emitStream = [];
        private int _emitCursor;
        private int _playbackQueueOffset;
        private float _currentPlaybackGain = 1;
        private long _lastRealCommitIndex;
        private long _duckCommitIndex = -1;
        private long _drainTargetIndex = -1;
        private int _restartSilenceRemaining;
        private int _pendingCaptureDiscontinuity;
        private bool _playbackCompletionPending;
        private long _streamEpoch;
        private long _sequence;
        private long _capturePosition;
        private long _renderStreamEpoch;
        private long _renderPosition;
        private bool _disposed;
        private bool _playbackHeld;

        public BenchmarkDuplexAudioIngress(
            IAcousticEchoCancellationService aecService,
            TimeProvider timeProvider,
            uint sampleRate = 16000,
            uint channels = 1)
        {
            _aecService = aecService ?? throw new ArgumentNullException(nameof(aecService));
            _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
            if (sampleRate == 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate));
            if (channels == 0)
                throw new ArgumentOutOfRangeException(nameof(channels));

            Options = new BenchmarkAudioOptions
            {
                SampleRate = sampleRate,
                ChannelCount = channels,
                SampleFormat = SampleFormat.Float32,
            };
        }

        public AudioOptionsBase Options { get; }
        public bool IsRecording { get; private set; }
        public bool IsPlaying => _playbackHeld;
        public int CurrentQueueSize
        {
            get
            {
                lock (_playbackCaptureSync)
                    return Math.Max(0, _playbackQueue.Count - _playbackQueueOffset);
            }
        }
        public AudioServiceState State { get; private set; } = AudioServiceState.Idle;
        public bool EnableAec { get; set; } = true;
        public bool EnableDnr { get; set; }
        public bool EnableHighPassFilter { get; set; }
        public float PlaybackGain { get; private set; } = 1;

        /// <summary>
        /// Committed output latency: how much audio the device has accepted but not yet emitted.
        /// Neither a gain change nor a queue clear can affect it. Defaults to 20 ms to match the
        /// shipping <c>SuggestedLatency</c> of the PortAudio duplex service. Set to 0 to restore the
        /// old zero-latency behaviour.
        /// </summary>
        public int DeviceOutputLatencySamples { get; set; }

        /// <summary>
        /// Ramp applied to playback gain changes, mirroring the real service's callback-side ramp.
        /// </summary>
        public float PlaybackGainRampMs { get; set; } = 15f;

        /// <summary>
        /// Silence emitted after a queue clear to represent the device stream stop/restart window.
        /// Defaults to 0: once the queue is cleared and the committed buffer has drained there is
        /// nothing left to emit anyway, so the restart's output-side duration does not change what
        /// the user hears. Exposed for sweeping.
        /// </summary>
        public int StreamRestartSilenceSamples { get; set; }

        /// <summary>
        /// When set, a queue clear also restarts the duplex stream, which breaks the capture
        /// timeline and resets the echo canceller. This is the capture-side cost of a barge-in stop
        /// on the real service and it lands on the frames carrying the barge-in utterance, so it is
        /// modelled independently of the output-side restart duration.
        /// </summary>
        public bool ModelStreamRestartOnClear { get; set; }

        public long PlaybackDuckCapturePosition { get; private set; } = -1;
        public long PlaybackClearCapturePosition { get; private set; } = -1;
        public long PlaybackResumeCapturePosition { get; private set; } = -1;
        public long PlaybackAudibleStopCapturePosition { get; private set; } = -1;

        /// <summary>
        /// Capture position at which the loudspeaker first emitted attenuated audio. Later than
        /// <see cref="PlaybackDuckCapturePosition"/> by the committed output latency plus the gain
        /// ramp; this is the one that corresponds to the user actually hearing the duck.
        /// </summary>
        public long PlaybackAudibleDuckCapturePosition { get; private set; } = -1;

        /// <summary>
        /// Capture position at which the committed device buffer finished draining after a queue
        /// clear, i.e. the last moment the loudspeaker emitted assistant audio. Excludes room
        /// reverberation, which <see cref="PlaybackAudibleStopCapturePosition"/> covers.
        /// </summary>
        public long PlaybackDrainCompleteCapturePosition { get; private set; } = -1;
        public long CurrentCaptureSamplePosition => Volatile.Read(ref _capturePosition);
        public int CapturedPlaybackSampleCount
        {
            get
            {
                lock (_playbackCaptureSync)
                    return _capturedPlayback.Count;
            }
        }
        public int CaptureProcessorDelaySamples
            => EnableAec
                ? Math.Max(0, _aecService.DelaySamples)
                : 0;
        public event Action<ReadOnlySpan<float>>? OnDataAvailable;
        public event EventHandler? OnPlaybackCompleted;
        public event EventHandler<AudioQueueOverflowEventArgs>? OnQueueOverflow
        {
            add { }
            remove { }
        }
        public event EventHandler<AudioErrorEventArgs>? OnError
        {
            add { }
            remove { }
        }
        public event EventHandler<AudioXrunEventArgs>? OnXrun
        {
            add { }
            remove { }
        }
        public event DuplexAudioFrameHandler? OnDuplexAudioFrame;
        public event ProcessedCaptureFrameHandler? OnProcessedCaptureFrame;

        public IReadOnlyList<AudioDevice> ListAudioDevices(
            AudioDeviceType? deviceType = null,
            bool enumAllDevices = false)
            => [];

        public int GetDefaultInputDevice() => 0;
        public int GetDefaultOutputDevice() => 0;

        public Task StartAsync(
            int inputDeviceIndex = -1,
            int outputDeviceIndex = -1,
            CancellationToken cancellationToken = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            cancellationToken.ThrowIfCancellationRequested();
            _aecService.Reset();
            _streamEpoch = checked(_streamEpoch + 1);
            _sequence = 0;
            _capturePosition = 0;
            _renderStreamEpoch = checked(_renderStreamEpoch + 1);
            _renderPosition = 0;
            _captureTimelineMapper.Reset();
            _playbackHeld = false;
            PlaybackGain = 1;
            PlaybackDuckCapturePosition = -1;
            PlaybackClearCapturePosition = -1;
            PlaybackResumeCapturePosition = -1;
            PlaybackAudibleStopCapturePosition = -1;
            PlaybackAudibleDuckCapturePosition = -1;
            PlaybackDrainCompleteCapturePosition = -1;
            Volatile.Write(ref _pendingCaptureDiscontinuity, 0);
            lock (_playbackCaptureSync)
            {
                _capturedPlayback.Clear();
                _playbackQueue.Clear();
                _playbackQueueOffset = 0;
                _playbackCompletionPending = false;

                // The device buffer starts pre-filled with silence, so the first committed samples
                // become audible one full output latency later, as on real hardware.
                _emitStream.Clear();
                for (int i = 0; i < DeviceOutputLatencySamples; i++)
                    _emitStream.Add(0);
                _emitCursor = 0;
                _currentPlaybackGain = 1;
                _lastRealCommitIndex = 0;
                _duckCommitIndex = -1;
                _drainTargetIndex = -1;
                _restartSilenceRemaining = 0;
            }
            IsRecording = true;
            State = AudioServiceState.Active;
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            IsRecording = false;
            _playbackHeld = false;
            _playbackCompletionPending = false;
            PlaybackGain = 1;
            State = AudioServiceState.Idle;
            return Task.CompletedTask;
        }

        public Task StartRecordingAsync(
            int deviceIndex = -1,
            CancellationToken cancellationToken = default)
            => StartAsync(deviceIndex, -1, cancellationToken);

        public Task StopRecordingAsync(CancellationToken cancellationToken = default)
            => StopAsync(cancellationToken);

        public Task StartPlayingAsync(
            int deviceIndex = -1,
            CancellationToken cancellationToken = default)
            => StartAsync(-1, deviceIndex, cancellationToken);

        public Task StopPlayingAsync(CancellationToken cancellationToken = default)
            => StopAsync(cancellationToken);

        public Task SwitchDeviceAsync(int deviceIndex, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.CompletedTask;
        }

        public bool TryEnqueue(ReadOnlySpan<float> data, out int droppedSamples)
        {
            if (State != AudioServiceState.Active)
            {
                droppedSamples = data.Length;
                return false;
            }

            _playbackHeld = true;
            lock (_playbackCaptureSync)
            {
                foreach (float sample in data)
                {
                    _capturedPlayback.Add(sample);
                    _playbackQueue.Add(sample);
                }
                _playbackCompletionPending = false;
            }
            droppedSamples = 0;
            return true;
        }

        /// <summary>
        /// Clears queued playback but lets audio already sent to the device drain, matching the
        /// delay the user hears after a stop. The benchmark omits the blocked control thread and
        /// dropped capture samples, but models the restart's AEC reset and timeline discontinuity.
        /// </summary>
        public void ClearQueue()
        {
            bool wasPlaying;
            lock (_playbackCaptureSync)
            {
                wasPlaying = HasAudioInFlight();
                _playbackQueue.Clear();
                _playbackQueueOffset = 0;

                // Everything already handed to the device still has to come out of the speaker.
                _drainTargetIndex = _lastRealCommitIndex;
                if (_emitCursor >= _drainTargetIndex)
                {
                    // Nothing was in flight, so the drain is already complete.
                    _playbackCompletionPending = false;
                    _playbackHeld = false;
                    if (PlaybackDrainCompleteCapturePosition < 0)
                        PlaybackDrainCompleteCapturePosition = _capturePosition;
                }
                else
                {
                    // Keep the completion armed so CommitPlaybackFrame raises it once the committed
                    // buffer has finished draining, rather than reporting completion while the
                    // loudspeaker is still emitting.
                    _playbackCompletionPending = true;
                    if (StreamRestartSilenceSamples > 0)
                        _restartSilenceRemaining = StreamRestartSilenceSamples;
                }

                if (ModelStreamRestartOnClear)
                    Volatile.Write(ref _pendingCaptureDiscontinuity, 1);
            }

            PlaybackClearCapturePosition = _capturePosition;
            if (wasPlaying && !_playbackHeld)
                OnPlaybackCompleted?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Requests a playback gain change. The new gain is reached over
        /// <see cref="PlaybackGainRampMs"/> and applies only to samples not yet committed to the
        /// device, mirroring the real service's ramped callback-side gain.
        /// </summary>
        public void SetPlaybackGain(float gain)
        {
            float clamped = Math.Clamp(gain, 0, 1);
            if (clamped < PlaybackGain && PlaybackDuckCapturePosition < 0)
                PlaybackDuckCapturePosition = _capturePosition;
            if (clamped > PlaybackGain &&
                PlaybackDuckCapturePosition >= 0 &&
                PlaybackResumeCapturePosition < 0)
            {
                PlaybackResumeCapturePosition = _capturePosition;
            }
            PlaybackGain = clamped;
        }

        /// <summary>
        /// Returns the samples the loudspeaker emits during this callback. Newly dequeued samples
        /// are pushed through a committed-output delay line of
        /// <see cref="DeviceOutputLatencySamples"/>, so a gain change or a queue clear issued now
        /// cannot affect audio the device has already accepted.
        /// </summary>
        public void ReadPlayback(Span<float> destination)
        {
            destination.Clear();
            lock (_playbackCaptureSync)
            {
                AppendToDeviceBuffer(destination.Length);

                int count = Math.Min(destination.Length, _emitStream.Count - _emitCursor);
                for (int i = 0; i < count; i++)
                    destination[i] = _emitStream[_emitCursor + i];
                _emitCursor += count;

                if (_duckCommitIndex >= 0 &&
                    PlaybackAudibleDuckCapturePosition < 0 &&
                    _emitCursor > _duckCommitIndex)
                {
                    PlaybackAudibleDuckCapturePosition = _capturePosition;
                }
                if (_drainTargetIndex >= 0 &&
                    PlaybackDrainCompleteCapturePosition < 0 &&
                    _emitCursor >= _drainTargetIndex)
                {
                    PlaybackDrainCompleteCapturePosition = _capturePosition;
                }
            }
        }

        /// <summary>
        /// Moves up to <paramref name="count"/> samples from the software queue into the committed
        /// device buffer, applying the ramped gain. Underruns are padded with silence, which is what
        /// the device callback does when the application cannot keep up.
        /// </summary>
        private void AppendToDeviceBuffer(int count)
        {
            float target = PlaybackGain;
            float step = PlaybackGainRampMs > 0 && Options.SampleRate > 0
                ? 1f / (PlaybackGainRampMs * Options.SampleRate / 1000f)
                : 1f;

            for (int i = 0; i < count; i++)
            {
                if (_currentPlaybackGain < target)
                    _currentPlaybackGain = Math.Min(target, _currentPlaybackGain + step);
                else if (_currentPlaybackGain > target)
                    _currentPlaybackGain = Math.Max(target, _currentPlaybackGain - step);

                if (_restartSilenceRemaining > 0)
                {
                    _restartSilenceRemaining--;
                    _emitStream.Add(0);
                    continue;
                }

                int available = _playbackQueue.Count - _playbackQueueOffset;
                if (available <= 0)
                {
                    // Device underrun: the callback still has to emit something.
                    _emitStream.Add(0);
                    continue;
                }

                float sample = _playbackQueue[_playbackQueueOffset++] * _currentPlaybackGain;
                _emitStream.Add(sample);
                _lastRealCommitIndex = _emitStream.Count;
                if (_currentPlaybackGain < 1f && _duckCommitIndex < 0)
                    _duckCommitIndex = _emitStream.Count - 1;

                if (_playbackQueueOffset >= _playbackQueue.Count)
                {
                    _playbackQueue.Clear();
                    _playbackQueueOffset = 0;
                    _playbackCompletionPending = true;
                }
            }
        }

        private bool HasAudioInFlight()
            => _playbackQueue.Count - _playbackQueueOffset > 0 ||
               _emitCursor < _lastRealCommitIndex;

        public void CommitPlaybackFrame()
        {
            bool completed = false;
            lock (_playbackCaptureSync)
            {
                // Playback is only finished once the committed device buffer has drained as well,
                // not when the software queue empties.
                if (_playbackCompletionPending &&
                    _playbackQueue.Count == 0 &&
                    _emitCursor >= _lastRealCommitIndex)
                {
                    _playbackCompletionPending = false;
                    _playbackHeld = false;
                    completed = true;
                }
            }
            if (completed)
                OnPlaybackCompleted?.Invoke(this, EventArgs.Empty);
        }

        public void MarkPlaybackAudibleStop(long capturePosition)
        {
            if (PlaybackClearCapturePosition < 0 ||
                PlaybackAudibleStopCapturePosition >= 0)
            {
                return;
            }
            PlaybackAudibleStopCapturePosition = Math.Max(
                PlaybackClearCapturePosition,
                capturePosition);
        }

        public float[] CopyCapturedPlayback()
        {
            lock (_playbackCaptureSync)
                return _capturedPlayback.ToArray();
        }

        public CaptureRange PushFrame(
            ReadOnlySpan<float> microphone,
            ReadOnlySpan<float> render,
            bool hasDiscontinuity = false,
            RenderTimelineStatus renderStatus = RenderTimelineStatus.Valid,
            bool hasRenderDiscontinuity = false)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (State != AudioServiceState.Active)
                throw new InvalidOperationException("Benchmark duplex ingress is not active.");
            if (microphone.Length != render.Length)
                throw new ArgumentException("Microphone and render frames must have the same length.");
            if (microphone.IsEmpty)
                throw new ArgumentException("A benchmark frame cannot be empty.", nameof(microphone));
            if (microphone.Length % Options.ChannelCount != 0)
                throw new ArgumentException("Interleaved frame length must be divisible by the channel count.");
            if (renderStatus is not RenderTimelineStatus.Valid and
                not RenderTimelineStatus.Gap and
                not RenderTimelineStatus.Missing)
            {
                throw new ArgumentOutOfRangeException(nameof(renderStatus));
            }
            if (renderStatus == RenderTimelineStatus.Gap && hasRenderDiscontinuity)
            {
                throw new ArgumentException(
                    "A render gap must remain on the current render timeline.",
                    nameof(hasRenderDiscontinuity));
            }

            if (hasRenderDiscontinuity)
            {
                _renderStreamEpoch = checked(_renderStreamEpoch + 1);
                _renderPosition = 0;
            }

            long renderStart = _renderPosition;
            _renderPosition = checked(_renderPosition + render.Length);
            var renderTimeline = new RenderSignalTimeline(
                _renderStreamEpoch,
                renderStart,
                _renderPosition,
                renderStatus,
                hasRenderDiscontinuity);

            // A device stream restart caused by a barge-in stop breaks the capture timeline and
            // forces an AEC reset, and it lands exactly where the barge-in utterance is being
            // recorded. Fold it into this frame's discontinuity so the production path sees it the
            // same way it would on hardware.
            if (Interlocked.Exchange(ref _pendingCaptureDiscontinuity, 0) != 0)
                hasDiscontinuity = true;

            float[] raw = microphone.ToArray();
            float[] beforeAec = raw.ToArray();
            float[] afterAec = beforeAec.ToArray();
            float[] renderCopy = render.ToArray();
            int captureProcessorDelay = CaptureProcessorDelaySamples;
            if (hasDiscontinuity)
                _aecService.Reset();
            if (EnableAec)
            {
                _aecService.Process(
                    afterAec,
                    renderCopy,
                    Options.ChannelCount,
                    Options.SampleRate);
            }

            long start = _capturePosition;
            long sequence = checked(++_sequence);
            _capturePosition = checked(_capturePosition + afterAec.Length);
            long timestamp = _timeProvider.GetTimestamp();
            var range = new CaptureRange(
                _streamEpoch,
                sequence,
                sequence,
                start,
                _capturePosition,
                timestamp);
            BufferedCaptureOutput captureOutput = _captureTimelineMapper.Map(
                in range,
                afterAec.Length,
                buffered: EnableAec && captureProcessorDelay > 0,
                captureProcessorDelay,
                hasDiscontinuity);
            CaptureSignalTimeline callbackTimeline =
                CaptureSignalTimeline.FromWholeBuffer(in range, afterAec.Length);
            var timeline = new DuplexCaptureTimeline(
                callbackTimeline,
                callbackTimeline,
                captureOutput.Timeline);
            var duplex = new DuplexAudioFrame(
                SequenceNumber: sequence,
                TimestampTicks: timestamp,
                SampleRate: Options.SampleRate,
                Channels: Options.ChannelCount,
                FrameCount: (uint)(afterAec.Length / Options.ChannelCount),
                CaptureProcessed: EnableAec,
                RenderReference: renderCopy,
                CaptureRaw: raw,
                CaptureBeforeAec: beforeAec,
                CaptureAfterAec: afterAec,
                StreamEpoch: _streamEpoch,
                StartSample: start,
                HasDiscontinuity: hasDiscontinuity || captureOutput.MappingReset,
                Timeline: timeline,
                RenderTimeline: renderTimeline,
                PlaybackGain: PlaybackGain);
            OnDuplexAudioFrame?.Invoke(in duplex);

            if (!captureOutput.IsValid)
                return range;

            var processedAudio = new ReadOnlyMemory<float>(
                afterAec,
                captureOutput.BufferOffset,
                captureOutput.SampleCount);
            var processed = new ProcessedCaptureFrame(
                captureOutput.SourceRange,
                Options.SampleRate,
                Options.ChannelCount,
                processedAudio,
                DelaySamples: checked((int)Math.Min(
                    int.MaxValue,
                    Math.Max(0, range.EndSampleExclusive - captureOutput.SourceRange.EndSampleExclusive))),
                HasDiscontinuity: hasDiscontinuity || captureOutput.HasDiscontinuity);
            OnProcessedCaptureFrame?.Invoke(in processed);
            OnDataAvailable?.Invoke(processedAudio.Span);
            return range;
        }

        public void FlushCaptureProcessorDelay()
        {
            int delaySamples = CaptureProcessorDelaySamples;
            if (delaySamples <= 0 || State != AudioServiceState.Active)
                return;

            float[] silence = new float[delaySamples];
            PushFrame(silence, silence);
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            IsRecording = false;
            _playbackHeld = false;
            _playbackCompletionPending = false;
            State = AudioServiceState.Idle;
        }

        private sealed class BenchmarkAudioOptions : AudioOptionsBase
        {
        }
    }
}
