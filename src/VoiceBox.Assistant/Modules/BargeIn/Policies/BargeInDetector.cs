using VoiceBox.Assistant.Modules.BargeIn.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies
{
    /// <summary>
    /// 维护 barge-in 候选音频及确认阈值。确认后会保留可识别音频，供新的 ASR session 补喂。
    /// </summary>
    internal sealed class BargeInDetector
    {
        private readonly List<float[]> _pendingAudio = new();
        private readonly int _sampleRate;
        private readonly int _channels;
        private readonly BargeInConfirmationSettings _defaultSettings;
        private long _minSamples;
        private float _minRms;
        private long _maxCandidateSamples;
        private long _recognitionAudioLimitSamples;
        private long _suppressionSamplesRemaining;
        private long _recognitionSuppressionSamplesRemaining;
        private long _pendingAudioSamples;
        private long _samples;
        private double _energy;
        private long _confirmationSamples;
        private double _confirmationEnergy;
        private bool _confirmationMet;

        public BargeInDetector(int minDurationMs, float minRms, int sampleRate, int channels)
        {
            _sampleRate = sampleRate;
            _channels = channels;
            _defaultSettings = new BargeInConfirmationSettings(minDurationMs, minRms);
            ApplySettings(_defaultSettings);
        }

        public bool IsCandidateActive { get; private set; }
        public bool IsConfirmationMet => _confirmationMet;
        public long CandidateSamples => _samples;
        public long ConfirmationSamples => _confirmationSamples;
        public long InterruptedTurnId { get; private set; }
        public CancellationTokenSource? ResponseCts { get; private set; }
        public IReadOnlyList<float[]> PendingAudio => _pendingAudio;
        public BargeInConfirmationSettings DefaultSettings => _defaultSettings;
        public BargeInConfirmationSettings CurrentSettings { get; private set; }
        public void BeginCandidate(long interruptedTurnId, CancellationTokenSource? responseCts)
            => BeginCandidate(interruptedTurnId, responseCts, _defaultSettings);

        public void BeginCandidate(
            long interruptedTurnId,
            CancellationTokenSource? responseCts,
            BargeInConfirmationSettings settings)
        {
            Clear();
            ApplySettings(settings);
            IsCandidateActive = true;
            InterruptedTurnId = interruptedTurnId;
            ResponseCts = responseCts;
        }

        public bool AddAudio(ReadOnlySpan<float> audio)
        {
            if (!IsCandidateActive || audio.IsEmpty)
                return false;

            if (_confirmationMet)
            {
                AddAudioAfterConfirmation(audio);
                return true;
            }

            for (int offset = 0; offset < audio.Length;)
            {
                if (_maxCandidateSamples > 0 && _samples >= _maxCandidateSamples)
                    RestartCandidateKeepingContext();

                int length = audio.Length - offset;
                if (_maxCandidateSamples > 0)
                {
                    long remaining = _maxCandidateSamples - _samples;
                    if (remaining <= 0)
                    {
                        RestartCandidateKeepingContext();
                        remaining = _maxCandidateSamples;
                    }

                    length = (int)Math.Min(length, remaining);
                }

                if (AddAudioCore(audio.Slice(offset, length)))
                {
                    _confirmationMet = true;
                    return true;
                }

                offset += length;
            }

            return false;
        }

        private void AddAudioAfterConfirmation(ReadOnlySpan<float> audio)
        {
            int length = audio.Length;
            if (_maxCandidateSamples > 0)
            {
                long remaining = _maxCandidateSamples - _samples;
                if (remaining <= 0)
                    return;
                length = (int)Math.Min(length, remaining);
            }

            ReadOnlySpan<float> retained = audio[..length];
            AddPendingAudio(retained, trimToRecentLimit: false);
            for (int i = 0; i < retained.Length; i++)
            {
                float sample = retained[i];
                _energy += (double)sample * sample;
            }
            _samples += retained.Length;
        }

        private bool AddAudioCore(ReadOnlySpan<float> audio)
        {
            if (audio.IsEmpty)
                return false;

            int recognitionOffset = 0;
            if (_recognitionSuppressionSamplesRemaining > 0)
            {
                recognitionOffset = (int)Math.Min(audio.Length, _recognitionSuppressionSamplesRemaining);
                _recognitionSuppressionSamplesRemaining -= recognitionOffset;
            }

            AddPendingAudio(audio[recognitionOffset..], trimToRecentLimit: true);

            for (int i = 0; i < audio.Length; i++)
            {
                float sample = audio[i];
                _energy += (double)sample * sample;
            }
            _samples += audio.Length;

            int confirmationOffset = 0;
            if (_suppressionSamplesRemaining > 0)
            {
                confirmationOffset = (int)Math.Min(audio.Length, _suppressionSamplesRemaining);
                _suppressionSamplesRemaining -= confirmationOffset;
            }

            ReadOnlySpan<float> confirmationAudio = audio[confirmationOffset..];
            for (int i = 0; i < confirmationAudio.Length; i++)
            {
                float sample = confirmationAudio[i];
                _confirmationEnergy += (double)sample * sample;
            }
            _confirmationSamples += confirmationAudio.Length;

            return MeetsThreshold();
        }

        public void MarkConfirmedKeepingPendingAudio()
        {
            IsCandidateActive = false;
            _samples = 0;
            _energy = 0;
            _confirmationSamples = 0;
            _confirmationEnergy = 0;
            _suppressionSamplesRemaining = 0;
            _recognitionSuppressionSamplesRemaining = 0;
            _maxCandidateSamples = 0;
            _recognitionAudioLimitSamples = 0;
            _confirmationMet = false;
            InterruptedTurnId = 0;
            ResponseCts = null;
        }

        public void Clear()
        {
            _pendingAudio.Clear();
            _pendingAudioSamples = 0;
            IsCandidateActive = false;
            _samples = 0;
            _energy = 0;
            _confirmationSamples = 0;
            _confirmationEnergy = 0;
            _suppressionSamplesRemaining = 0;
            _recognitionSuppressionSamplesRemaining = 0;
            _maxCandidateSamples = 0;
            _recognitionAudioLimitSamples = 0;
            _confirmationMet = false;
            InterruptedTurnId = 0;
            ResponseCts = null;
        }

        public void ClearPendingAudio()
        {
            _pendingAudio.Clear();
            _pendingAudioSamples = 0;
        }

        public double ComputeRms()
        {
            if (_samples <= 0)
                return 0;
            return Math.Sqrt(_energy / _samples);
        }

        public double ComputeConfirmationRms()
        {
            if (_confirmationSamples <= 0)
                return 0;
            return Math.Sqrt(_confirmationEnergy / _confirmationSamples);
        }

        private void ApplySettings(BargeInConfirmationSettings settings)
        {
            CurrentSettings = settings;
            _minSamples = SamplesFromMilliseconds(settings.MinDurationMs);
            _minRms = settings.MinRms;
            _maxCandidateSamples = SamplesFromMilliseconds(settings.MaxCandidateMs);
            _recognitionAudioLimitSamples = SamplesFromMilliseconds(settings.RecognitionAudioLimitMs);
            _suppressionSamplesRemaining = SamplesFromMilliseconds(settings.InitialSuppressionMs);
            _recognitionSuppressionSamplesRemaining = SamplesFromMilliseconds(settings.RecognitionSuppressionMs);
        }

        private long SamplesFromMilliseconds(int milliseconds)
        {
            return _sampleRate > 0 && _channels > 0
                ? (long)milliseconds * _sampleRate * _channels / 1000
                : 0;
        }

        private void AddPendingAudio(ReadOnlySpan<float> audio, bool trimToRecentLimit)
        {
            if (audio.IsEmpty)
                return;

            float[] copy = audio.ToArray();
            _pendingAudio.Add(copy);
            _pendingAudioSamples += copy.Length;
            if (trimToRecentLimit)
                TrimPendingAudioToLimit();
        }

        private void TrimPendingAudioToLimit()
        {
            if (_recognitionAudioLimitSamples <= 0)
                return;

            long overflow = _pendingAudioSamples - _recognitionAudioLimitSamples;
            while (overflow > 0 && _pendingAudio.Count > 0)
            {
                float[] first = _pendingAudio[0];
                if (overflow >= first.Length)
                {
                    _pendingAudio.RemoveAt(0);
                    _pendingAudioSamples -= first.Length;
                    overflow -= first.Length;
                    continue;
                }

                int keep = first.Length - (int)overflow;
                var trimmed = new float[keep];
                Array.Copy(first, first.Length - keep, trimmed, 0, keep);
                _pendingAudio[0] = trimmed;
                _pendingAudioSamples -= overflow;
                overflow = 0;
            }
        }

        private void RestartCandidateKeepingContext()
        {
            long interruptedTurnId = InterruptedTurnId;
            CancellationTokenSource? responseCts = ResponseCts;
            BargeInConfirmationSettings settings = CurrentSettings;

            Clear();
            BeginCandidate(interruptedTurnId, responseCts, settings);
        }

        private bool MeetsThreshold()
        {
            if (_suppressionSamplesRemaining > 0 || _confirmationSamples <= 0)
                return false;

            if (_confirmationSamples < _minSamples)
                return false;

            return ComputeConfirmationRms() >= _minRms;
        }
    }
}
