using System.Diagnostics;
using VoiceBox.AEC.DFSMN.Models;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class NearendVadStateMachine
    {
        private enum VadState { Silence, PossibleSpeech, Speech, PossibleSilence }
        private enum VadEventType { SpeechStart, SpeechEnd }

        private readonly struct VadEvent
        {
            public VadEvent(VadEventType type, long startHop, long endHop)
            {
                Type = type;
                StartHop = startHop;
                EndHop = endHop;
            }

            public VadEventType Type { get; }
            public long StartHop { get; }
            public long EndHop { get; }
        }

        private readonly DfsmnVadOptions _options;
        private readonly int _hopSamples;
        private readonly int _preSpeechPaddingHops;
        private readonly float[] _smoothRing;
        private int _smoothWrite;
        private int _smoothCount;
        private float _smoothSum;

        private readonly float[] _audioRing;
        private readonly int _audioRingCapacityHops;
        private long _audioRingWriteHop = 1;
        private long _hopCount;
        private VadState _state = VadState.Silence;
        private int _speechHopCount;
        private int _silenceHopCount;
        private bool _hitMaxSpeech;
        private long _lastSpeechStartHop = -1;
        private long _lastSpeechEndHop;

        private bool _isEmitting;
        private readonly List<float> _speechBuffer;
        private long _lastEmittedHop;
        private readonly Stopwatch _stopwatch = new();

        public NearendVadStateMachine(DfsmnVadOptions options, int hopSamples, int sampleRate)
        {
            ArgumentNullException.ThrowIfNull(options);
            options.Validate();
            if (hopSamples <= 0) throw new ArgumentOutOfRangeException(nameof(hopSamples));
            if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));

            _options = options;
            _hopSamples = hopSamples;
            _smoothRing = new float[_options.SmoothWindowHops];

            _preSpeechPaddingHops = (int)Math.Ceiling(_options.PreSpeechPaddingMs / (1000.0 * hopSamples / sampleRate));
            _audioRingCapacityHops = Math.Max(
                8,
                Math.Max(_options.MaxSpeechHops, _options.MinSpeechHops + _options.MinSilenceHops) +
                _options.PadStartHops + _preSpeechPaddingHops + _options.MinSpeechHops + _options.MinSilenceHops + 8);
            _audioRing = new float[_audioRingCapacityHops * hopSamples];
            _speechBuffer = new List<float>(sampleRate * 10);
        }

        public NearendVadProbabilityObservation ProcessHop(float vadProbability, ReadOnlySpan<float> audioHop, Action<NearendVadNotification> notify)
        {
            ArgumentNullException.ThrowIfNull(notify);
            if (audioHop.Length != _hopSamples)
                throw new ArgumentException($"audioHop length must be {_hopSamples}.", nameof(audioHop));

            if (!float.IsFinite(vadProbability))
                vadProbability = 0f;
            vadProbability = Math.Clamp(vadProbability, 0f, 1f);

            long hopIndex = _hopCount + 1;
            WriteAudioHop(hopIndex, audioHop);

            VadEvent? evt = ProcessProbability(vadProbability, out NearendVadProbabilityObservation observation);
            if (evt.HasValue)
                HandleEvent(evt.Value, notify);

            if (_isEmitting && _lastEmittedHop < _hopCount)
                EmitAudio(_lastEmittedHop + 1, _hopCount, notify);

            return observation;
        }

        public void Reset()
        {
            Array.Clear(_smoothRing);
            _smoothWrite = 0;
            _smoothCount = 0;
            _smoothSum = 0f;

            Array.Clear(_audioRing);
            _audioRingWriteHop = 1;
            _hopCount = 0;
            _state = VadState.Silence;
            _speechHopCount = 0;
            _silenceHopCount = 0;
            _hitMaxSpeech = false;
            _lastSpeechStartHop = -1;
            _lastSpeechEndHop = 0;

            _isEmitting = false;
            _speechBuffer.Clear();
            _lastEmittedHop = 0;
            _stopwatch.Reset();
        }

        private VadEvent? ProcessProbability(float probability, out NearendVadProbabilityObservation observation)
        {
            _hopCount++;

            if (_smoothCount == _smoothRing.Length)
                _smoothSum -= _smoothRing[_smoothWrite];
            else
                _smoothCount++;

            _smoothRing[_smoothWrite] = probability;
            _smoothSum += probability;
            _smoothWrite = (_smoothWrite + 1) % _smoothRing.Length;

            float smoothed = _smoothSum / _smoothCount;
            bool isSpeech = smoothed >= _options.SpeechThreshold;
            observation = new NearendVadProbabilityObservation(probability, smoothed, isSpeech, _hopCount);

            VadEvent? evt = null;
            if (_hitMaxSpeech)
            {
                _hitMaxSpeech = false;
                if (isSpeech)
                {
                    _lastSpeechStartHop = _hopCount;
                    _state = VadState.Speech;
                    _speechHopCount = 1;
                    _silenceHopCount = 0;
                    return new VadEvent(VadEventType.SpeechStart, _hopCount, _hopCount);
                }

                _state = VadState.Silence;
                _speechHopCount = 0;
                _silenceHopCount = 1;
                return null;
            }

            switch (_state)
            {
                case VadState.Silence:
                    if (isSpeech)
                    {
                        _speechHopCount = 1;
                        if (_speechHopCount >= _options.MinSpeechHops)
                        {
                            _state = VadState.Speech;
                            int startPaddingHops = _options.PadStartHops + _preSpeechPaddingHops;
                            long start = Math.Max(1L, _hopCount - _speechHopCount + 1 - startPaddingHops);
                            if (_lastSpeechEndHop + 1 > start) start = _lastSpeechEndHop + 1;
                            _lastSpeechStartHop = start;
                            _silenceHopCount = 0;
                            evt = new VadEvent(VadEventType.SpeechStart, start, _hopCount);
                        }
                        else
                        {
                            _state = VadState.PossibleSpeech;
                        }
                    }
                    else
                    {
                        _silenceHopCount++;
                        _speechHopCount = 0;
                    }
                    break;

                case VadState.PossibleSpeech:
                    if (isSpeech)
                    {
                        _speechHopCount++;
                        if (_speechHopCount >= _options.MinSpeechHops)
                        {
                            _state = VadState.Speech;
                            int startPaddingHops = _options.PadStartHops + _preSpeechPaddingHops;
                            long start = Math.Max(1L, _hopCount - _speechHopCount + 1 - startPaddingHops);
                            if (_lastSpeechEndHop + 1 > start) start = _lastSpeechEndHop + 1;
                            _lastSpeechStartHop = start;
                            _silenceHopCount = 0;
                            evt = new VadEvent(VadEventType.SpeechStart, start, _hopCount);
                        }
                    }
                    else
                    {
                        _state = VadState.Silence;
                        _silenceHopCount = 1;
                        _speechHopCount = 0;
                    }
                    break;

                case VadState.Speech:
                    _speechHopCount++;
                    if (isSpeech)
                    {
                        _silenceHopCount = 0;
                        if (_options.MaxSpeechHops > 0 && _speechHopCount >= _options.MaxSpeechHops)
                            evt = ForceEndAtCurrentHop();
                    }
                    else
                    {
                        _state = VadState.PossibleSilence;
                        _silenceHopCount = 1;
                    }
                    break;

                case VadState.PossibleSilence:
                    _speechHopCount++;
                    if (isSpeech)
                    {
                        _state = VadState.Speech;
                        _silenceHopCount = 0;
                        if (_options.MaxSpeechHops > 0 && _speechHopCount >= _options.MaxSpeechHops)
                            evt = ForceEndAtCurrentHop();
                    }
                    else
                    {
                        _silenceHopCount++;
                        if (_silenceHopCount >= _options.MinSilenceHops)
                        {
                            _state = VadState.Silence;
                            evt = new VadEvent(VadEventType.SpeechEnd, _lastSpeechStartHop, _hopCount);
                            _lastSpeechEndHop = _hopCount;
                            _lastSpeechStartHop = -1;
                            _speechHopCount = 0;
                        }
                    }
                    break;
            }

            return evt;
        }

        private VadEvent ForceEndAtCurrentHop()
        {
            _hitMaxSpeech = true;
            _state = VadState.Silence;
            _speechHopCount = 0;
            var evt = new VadEvent(VadEventType.SpeechEnd, _lastSpeechStartHop, _hopCount);
            _lastSpeechEndHop = _hopCount;
            _lastSpeechStartHop = -1;
            return evt;
        }

        private void HandleEvent(VadEvent evt, Action<NearendVadNotification> notify)
        {
            switch (evt.Type)
            {
                case VadEventType.SpeechStart:
                    _speechBuffer.Clear();
                    _stopwatch.Restart();
                    _isEmitting = true;
                    _lastEmittedHop = Math.Max(0, evt.StartHop - 1);
                    notify(NearendVadNotification.SpeechStarted);
                    EmitAudio(evt.StartHop, evt.EndHop, notify);
                    break;

                case VadEventType.SpeechEnd:
                    if (!_isEmitting)
                        return;

                    if (_lastEmittedHop < evt.EndHop)
                        EmitAudio(_lastEmittedHop + 1, evt.EndHop, notify);

                    _stopwatch.Stop();
                    notify(new NearendVadNotification(
                        NearendVadNotificationKind.SpeechEnded,
                        _speechBuffer.ToArray(),
                        VADEventDataType.End,
                        _stopwatch.Elapsed));

                    _isEmitting = false;
                    _speechBuffer.Clear();
                    _lastEmittedHop = 0;
                    break;
            }
        }

        private void EmitAudio(long startHop, long endHop, Action<NearendVadNotification> notify)
        {
            float[] audio = ConcatHopAudio(startHop, endHop);
            if (audio.Length == 0)
            {
                _lastEmittedHop = Math.Max(_lastEmittedHop, endHop);
                return;
            }

            _speechBuffer.AddRange(audio);
            _lastEmittedHop = endHop;
            notify(new NearendVadNotification(
                NearendVadNotificationKind.Speaking,
                audio,
                VADEventDataType.Chunk));
        }

        private void WriteAudioHop(long hopIndex, ReadOnlySpan<float> audioHop)
        {
            int slot = (int)((hopIndex - 1) % _audioRingCapacityHops);
            audioHop.CopyTo(_audioRing.AsSpan(slot * _hopSamples, _hopSamples));
            _audioRingWriteHop = hopIndex + 1;
        }

        private float[] ConcatHopAudio(long startHop, long endHop)
        {
            if (startHop > endHop) return Array.Empty<float>();

            int writtenCount = (int)Math.Min(_audioRingWriteHop - 1, _audioRingCapacityHops);
            long oldestAvailable = Math.Max(1, _audioRingWriteHop - writtenCount);
            long newestAvailable = _audioRingWriteHop - 1;

            if (startHop < oldestAvailable) startHop = oldestAvailable;
            if (endHop > newestAvailable) endHop = newestAvailable;
            if (startHop > endHop) return Array.Empty<float>();

            int hopCount = (int)(endHop - startHop + 1);
            var result = new float[hopCount * _hopSamples];
            int startSlot = (int)((startHop - 1) % _audioRingCapacityHops);
            int slotsRemainingFromStart = _audioRingCapacityHops - startSlot;

            if (hopCount <= slotsRemainingFromStart)
            {
                Buffer.BlockCopy(_audioRing, startSlot * _hopSamples * sizeof(float), result, 0, result.Length * sizeof(float));
            }
            else
            {
                int firstSamples = slotsRemainingFromStart * _hopSamples;
                Buffer.BlockCopy(_audioRing, startSlot * _hopSamples * sizeof(float), result, 0, firstSamples * sizeof(float));
                int secondSamples = result.Length - firstSamples;
                Buffer.BlockCopy(_audioRing, 0, result, firstSamples * sizeof(float), secondSamples * sizeof(float));
            }

            return result;
        }
    }

    internal readonly record struct NearendVadProbabilityObservation(
        float Probability,
        float SmoothedProbability,
        bool IsSpeech,
        long HopIndex,
        CaptureRange CaptureRange = default);
}
