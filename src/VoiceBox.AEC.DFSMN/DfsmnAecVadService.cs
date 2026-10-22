using System.Diagnostics;
using Microsoft.Extensions.Options;
using VoiceBox.AEC.DFSMN.Internal;
using VoiceBox.AEC.DFSMN.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.AEC.DFSMN
{
    /// <summary>
    /// Combined DFSMN AEC + near-end VAD service. One model invocation per ready hop produces
    /// both the AEC spectral mask and the near-end VAD probability.
    /// </summary>
    public sealed class DfsmnAecVadService :
        IAcousticEchoCancellationService,
        IAcousticProcessingMetricsSource,
        IVoiceActivityDetectService
    {
        private readonly object _lock = new();
        private readonly DfsmnAecVadOptions _options;
        private readonly DfsmnAecPipeline _pipeline;
        private readonly DfsmnVadObservationQueue _observations;
        private readonly NearendVadStateMachine _vadStateMachine;
        private readonly float[] _drainHop;
        private readonly List<NearendVadNotification> _notifications = new();
        private readonly List<NearendVadProbabilityObservation> _probabilityObservations = new();
        private readonly float[] _probabilitySmoothRing;
        private int _probabilitySmoothWrite;
        private int _probabilitySmoothCount;
        private float _probabilitySmoothSum;
        private long _probabilityHopCount;

        private bool _disposed;

        public DfsmnAecVadService(IOptions<DfsmnAecVadOptions> options)
        {
            ArgumentNullException.ThrowIfNull(options);
            _options = options.Value
                ?? throw new ArgumentException("DFSMN AEC+VAD options value cannot be null.", nameof(options));
            _options.Validate();

            _pipeline = new DfsmnAecPipeline(_options, maskRunner: null, mvn: null, requireVadOutput: true);
            _observations = CreateObservationQueue(_options, _pipeline.HopLength);
            _vadStateMachine = new NearendVadStateMachine(_options.Vad, _pipeline.HopLength, (int)_options.SampleRate);
            _drainHop = new float[_pipeline.HopLength];
            _probabilitySmoothRing = new float[_options.Vad.SmoothWindowHops];
        }

        internal DfsmnAecVadService(IOptions<DfsmnAecVadOptions> options, IDfsmnMaskRunner runner, MvnTransform mvn)
        {
            ArgumentNullException.ThrowIfNull(options);
            ArgumentNullException.ThrowIfNull(runner);
            ArgumentNullException.ThrowIfNull(mvn);

            _options = options.Value
                ?? throw new ArgumentException("DFSMN AEC+VAD options value cannot be null.", nameof(options));
            _options.Validate();

            _pipeline = new DfsmnAecPipeline(_options, runner, mvn, requireVadOutput: true);
            _observations = CreateObservationQueue(_options, _pipeline.HopLength);
            _vadStateMachine = new NearendVadStateMachine(_options.Vad, _pipeline.HopLength, (int)_options.SampleRate);
            _drainHop = new float[_pipeline.HopLength];
            _probabilitySmoothRing = new float[_options.Vad.SmoothWindowHops];
        }

        public event EventHandler<EventArgs>? OnSpeechStarted;
        public event EventHandler<VADEventArgs>? OnSpeaking;
        public event EventHandler<VADEventArgs>? OnSpeechEnded;
        public event EventHandler<VadProbabilityEventArgs>? OnProbability;

        public int FrameSize => _pipeline.HopLength;

        public int DelaySamples => _pipeline.DelaySamples;

        public void Process(Span<float> capture, ReadOnlySpan<float> render, uint channels, uint sampleRate)
        {
            CaptureRange captureRange = default;
            Process(capture, render, channels, sampleRate, in captureRange);
        }

        public void Process(
            Span<float> capture,
            ReadOnlySpan<float> render,
            uint channels,
            uint sampleRate,
            in CaptureRange captureRange)
        {
            lock (_lock)
            {
                if (_disposed) return;

                DfsmnPipelineHopHandler? handler = HasVadSubscribers() ? OnPipelineHop : null;
                _pipeline.Process(capture, render, channels, sampleRate, in captureRange, handler);
            }
        }

        public void Predict(ReadOnlySpan<float> audioData)
        {
            NearendVadNotification[]? notifications = null;
            NearendVadProbabilityObservation[]? probabilityObservations = null;

            lock (_lock)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(DfsmnAecVadService));

                _notifications.Clear();
                _probabilityObservations.Clear();
                bool hasProbabilitySubscribers = OnProbability is not null;
                while (_observations.TryDequeue(
                    out float probability,
                    _drainHop,
                    out bool hasAudio,
                    out CaptureRange captureRange))
                {
                    NearendVadProbabilityObservation observation = ObserveProbability(
                        probability,
                        in captureRange);
                    if (hasAudio)
                        _vadStateMachine.ProcessHop(probability, _drainHop, _notifications.Add);

                    if (hasProbabilitySubscribers)
                        _probabilityObservations.Add(observation);
                }

                if (_notifications.Count > 0)
                {
                    notifications = _notifications.ToArray();
                    _notifications.Clear();
                }

                if (_probabilityObservations.Count > 0)
                {
                    probabilityObservations = _probabilityObservations.ToArray();
                    _probabilityObservations.Clear();
                }
            }

            if (probabilityObservations is not null)
                foreach (NearendVadProbabilityObservation observation in probabilityObservations)
                    RaiseProbability(observation);

            if (notifications is not null)
                foreach (NearendVadNotification notification in notifications)
                    RaiseVadNotification(notification);
        }

        void IAcousticEchoCancellationService.Reset() => ResetAecAndVad();

        void IVoiceActivityDetectService.Reset() => ResetVad();

        public bool TryGetLatestMetrics(out AcousticProcessingMetrics metrics)
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    metrics = default;
                    return false;
                }

                metrics = _pipeline.Metrics;
                return metrics.TimestampTicks != 0;
            }
        }

        public void ResetAecAndVad()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _pipeline.Reset();
                ResetVadLocked();
            }
        }

        public void ResetVad()
        {
            lock (_lock)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(DfsmnAecVadService));
                ResetVadLocked();
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _pipeline.Dispose();
                _observations.Clear();
                _probabilityObservations.Clear();
                _vadStateMachine.Reset();
            }
        }

        private static DfsmnVadObservationQueue CreateObservationQueue(DfsmnAecVadOptions options, int hopLength)
        {
            int preSpeechHops = (int)Math.Ceiling(options.Vad.PreSpeechPaddingMs / (1000.0 * hopLength / options.SampleRate));
            int capacity = Math.Max(64,
                Math.Max(options.Vad.MaxSpeechHops, options.Vad.MinSpeechHops + options.Vad.MinSilenceHops) +
                options.Vad.MinSilenceHops +
                options.Vad.PadStartHops +
                preSpeechHops +
                options.Vad.SmoothWindowHops +
                16);
            return new DfsmnVadObservationQueue(capacity, hopLength);
        }

        private bool HasVadSubscribers()
            => _options.Vad.Enabled && (
                HasSpeechSubscribers() ||
                OnProbability is not null);

        private bool HasSpeechSubscribers()
            => OnSpeechStarted is not null ||
                OnSpeaking is not null ||
                OnSpeechEnded is not null;

        private void OnPipelineHop(
            ReadOnlySpan<float> rawCapture,
            ReadOnlySpan<float> linearResidual,
            ReadOnlySpan<float> processedCapture,
            float vadProbability,
            in CaptureRange captureRange)
        {
            if (!HasSpeechSubscribers())
            {
                _observations.EnqueueProbability(vadProbability, in captureRange);
                return;
            }

            ReadOnlySpan<float> audio = _options.Vad.EventAudioSource switch
            {
                DfsmnVadAudioSource.RawCapture => rawCapture,
                DfsmnVadAudioSource.LinearResidual => linearResidual,
                _ => processedCapture,
            };

            _observations.Enqueue(vadProbability, audio, in captureRange);
        }

        private NearendVadProbabilityObservation ObserveProbability(
            float probability,
            in CaptureRange captureRange)
        {
            if (!float.IsFinite(probability))
                probability = 0f;
            probability = Math.Clamp(probability, 0f, 1f);

            _probabilityHopCount++;

            if (_probabilitySmoothCount == _probabilitySmoothRing.Length)
                _probabilitySmoothSum -= _probabilitySmoothRing[_probabilitySmoothWrite];
            else
                _probabilitySmoothCount++;

            _probabilitySmoothRing[_probabilitySmoothWrite] = probability;
            _probabilitySmoothSum += probability;
            _probabilitySmoothWrite = (_probabilitySmoothWrite + 1) % _probabilitySmoothRing.Length;

            float smoothed = _probabilitySmoothSum / _probabilitySmoothCount;
            bool isSpeech = smoothed >= _options.Vad.SpeechThreshold;
            return new NearendVadProbabilityObservation(
                probability,
                smoothed,
                isSpeech,
                _probabilityHopCount,
                captureRange);
        }

        private void RaiseVadNotification(NearendVadNotification notification)
        {
            switch (notification.Kind)
            {
                case NearendVadNotificationKind.SpeechStarted:
                    SafeInvoke(OnSpeechStarted, EventArgs.Empty);
                    break;
                case NearendVadNotificationKind.Speaking:
                    SafeInvoke(OnSpeaking, new VADEventArgs(notification.Audio, notification.DataType, notification.Duration));
                    break;
                case NearendVadNotificationKind.SpeechEnded:
                    SafeInvoke(OnSpeechEnded, new VADEventArgs(notification.Audio, notification.DataType, notification.Duration));
                    break;
            }
        }

        private void RaiseProbability(NearendVadProbabilityObservation observation)
        {
            EventHandler<VadProbabilityEventArgs>? handler = OnProbability;
            if (handler is null)
                return;

            double hopMs = 1000.0 * _pipeline.HopLength / _options.SampleRate;
            var timestamp = TimeSpan.FromMilliseconds((observation.HopIndex - 1) * hopMs);
            var frameDuration = TimeSpan.FromMilliseconds(hopMs);
            SafeInvoke(handler, new VadProbabilityEventArgs(
                observation.Probability,
                observation.SmoothedProbability,
                observation.IsSpeech,
                observation.HopIndex,
                timestamp,
                frameDuration,
                observation.CaptureRange));
        }

        private void ResetVadLocked()
        {
            _observations.Clear();
            _probabilityObservations.Clear();
            _vadStateMachine.Reset();
            ResetProbabilityOnlyLocked();
        }

        private void ResetProbabilityOnlyLocked()
        {
            Array.Clear(_probabilitySmoothRing);
            _probabilitySmoothWrite = 0;
            _probabilitySmoothCount = 0;
            _probabilitySmoothSum = 0f;
            _probabilityHopCount = 0;
        }

        private void SafeInvoke<T>(EventHandler<T>? handler, T args) where T : EventArgs
        {
            if (handler == null) return;
            try
            {
                handler.Invoke(this, args);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"VAD event handler failed: {ex.Message}");
            }
        }
    }

}
