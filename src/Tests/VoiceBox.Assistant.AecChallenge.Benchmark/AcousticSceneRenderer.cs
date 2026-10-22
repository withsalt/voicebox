using System.Numerics;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal static class AcousticSceneRenderer
{
    private const int SampleRate = EvaluationOptions.TargetSampleRate;
    private const int MaximumRirSamples = SampleRate * 2;
    private const int TailSamples = SampleRate * 2;
    private const int DriftLookaheadSamples = 64;

    /// <summary>
    /// Fixed near-end active speech level, -26 dBov. This is the anchor of the whole level model:
    /// it does not move with SER, SNR, geometry or corpus.
    /// </summary>
    public const double NearEndActiveSpeechLevel = 0.05;

    /// <summary>
    /// Loudspeaker active drive level, -14 dBov: the level the device actually plays TTS at. Held
    /// fixed so the loudspeaker nonlinearity is evaluated at a constant, bounded drive point instead
    /// of at whatever level the TTS decoder happened to emit.
    /// </summary>
    public const double SpeakerActiveDriveLevel = 0.2;

    public static RealtimeScene CreateRealtime(
        PreparedEndToEndCase testCase,
        float[] nominalRender,
        float[]? nearEnd,
        float[] echoRir,
        float[]? nearRir,
        float[]? noise,
        int callbackSamples,
        int startupGuardMs,
        int minimumSpeechMs,
        bool spatializeNoise)
    {
        if (callbackSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(callbackSamples));
        if (nominalRender.Length == 0)
            throw new InvalidDataException("The captured TTS render is empty.");
        if (testCase.OracleExpectHeard && (nearEnd is null || nearEnd.Length == 0))
            throw new InvalidDataException("A Heard-positive case requires near-end speech.");

        float[] echoImpulse = PrepareImpulse(echoRir);
        int echoDelaySamples = checked(testCase.EchoDelayMs * SampleRate / 1000);
        echoDelaySamples = Math.Max(0, echoDelaySamples - DriftLookaheadSamples);
        if (echoDelaySamples > 0)
            echoImpulse = Delay(echoImpulse, echoDelaySamples);

        // Level model.
        //
        // The near-end talker is the fixed anchor: a person does not raise their voice because the
        // assistant is loud. SER is therefore realized by moving the ECHO, which is what physically
        // happens when the device volume changes. The previous model did the opposite - it pinned
        // the echo and scaled the near-end by SER - which made the absolute near-end level a
        // function of SER. A fixed absolute threshold (BargeIn.MinRms, 0.03) then cut in below
        // 0 dB SER, so the low-SER end of the sweep measured whether the near-end signal cleared a
        // level floor rather than whether the detector could separate two overlapping talkers.
        double echoActiveTarget =
            NearEndActiveSpeechLevel / Math.Pow(10, testCase.SignalToEchoRatioDb / 20);

        // Loudspeaker drive level and echo return loss.
        //
        // Only two levels affect the microphone signal: the echo and the near-end. Both are now
        // determined - the near-end is the anchor and the echo follows from SER. What remains free is
        // how that echo level was produced: a loud speaker down a lossy path, or a quiet speaker
        // down a direct one. That choice decides how hard the loudspeaker is driven, and therefore
        // how much its nonlinearity distorts.
        //
        // The drive level is the quantity held fixed, because it is the one with a hard physical
        // bound - a device plays TTS near full scale and cannot exceed it. The echo return loss is
        // derived from it and reported. Deriving it the other way round (pinning ERL and solving for
        // drive) pushes the drive level past full scale at low SER and collapses every far-end
        // nonlinear case into pure tanh saturation, which would replace one confound with a worse
        // one.
        double speakerActiveTarget = testCase.EchoReturnLossDb is double pinnedEchoReturnLossDb
            ? echoActiveTarget * Math.Pow(10, pinnedEchoReturnLossDb / 20)
            : SpeakerActiveDriveLevel;
        double echoReturnLossDb = echoActiveTarget > 1e-12
            ? 20 * Math.Log10(speakerActiveTarget / echoActiveTarget)
            : 0;
        double renderActiveLevel = OracleActivityAnalyzer.ActiveSpeechLevel(
            nominalRender,
            SampleRate);
        double speakerGain = renderActiveLevel > 1e-9
            ? speakerActiveTarget / renderActiveLevel
            : 1;

        float[] nominalSpeaker = ApplyClockDrift(
            nominalRender,
            testCase.ClockDriftPpm);
        for (int i = 0; i < nominalSpeaker.Length; i++)
            nominalSpeaker[i] = (float)(nominalSpeaker[i] * speakerGain);
        if (testCase.FarEndNonlinear)
            ApplyLoudspeakerNonlinearity(nominalSpeaker);

        // Calibrate the room impulse so the echo lands on its target active level. The nominal echo
        // is produced through the same speaker-gain-then-nonlinearity path the realtime scene uses,
        // so the calibration remains valid after tanh has compressed the loud cases.
        float[] nominalEcho = Convolve(nominalSpeaker, echoImpulse);
        double nominalEchoActive = OracleActivityAnalyzer.ActiveSpeechLevel(
            nominalEcho,
            SampleRate);
        if (nominalEchoActive > 1e-9)
        {
            double echoScale = echoActiveTarget / nominalEchoActive;
            for (int i = 0; i < echoImpulse.Length; i++)
                echoImpulse[i] = (float)(echoImpulse[i] * echoScale);
        }

        int onset = checked(testCase.NearEndOnsetMs * SampleRate / 1000);
        float[] nearTrack = [];
        double nearActiveLevel = 0;
        if (nearEnd is { Length: > 0 })
        {
            float[] nearImpulse = nearRir is null ? [1] : PrepareImpulse(nearRir);
            float[] reverberantNear = Convolve(nearEnd, nearImpulse);
            ScaleToActiveSpeechLevel(reverberantNear, NearEndActiveSpeechLevel);
            nearActiveLevel = OracleActivityAnalyzer.ActiveSpeechLevel(
                reverberantNear,
                SampleRate);
            nearTrack = Delay(reverberantNear, onset);
        }

        int totalSamples = Math.Max(
            checked(nominalRender.Length + echoImpulse.Length),
            nearTrack.Length);
        totalSamples = checked(totalSamples + TailSamples);
        float[] noiseTrack = [];
        if (noise is { Length: > 0 })
        {
            noiseTrack = Repeat(noise, totalSamples);
            if (spatializeNoise)
            {
                // A dry external noise source (music, environment recordings) has to travel through
                // the room to reach the microphone, just like the talker does. Injecting it dry gave
                // it an unnaturally sharp spectrum and no reverberant tail - and this is exactly the
                // signal class that produces every false accept in this suite, so the realism of its
                // path decides whether those false accepts are real findings or artefacts.
                float[] noiseImpulse = nearRir is null ? [1] : PrepareImpulse(nearRir);
                if (noiseImpulse.Length > 1)
                {
                    float[] reverberantNoise = Convolve(noiseTrack, noiseImpulse);
                    noiseTrack = reverberantNoise.Length >= totalSamples
                        ? reverberantNoise[..totalSamples]
                        : reverberantNoise;
                }
            }

            // SNR is referenced to the fixed near-end anchor rather than to the realized near-end
            // track. Referencing the track would drag the noise level along with SER and make SNR a
            // second, hidden function of SER.
            double desiredNoiseRms =
                NearEndActiveSpeechLevel / Math.Pow(10, testCase.SignalToNoiseRatioDb / 20);
            ScaleToRms(noiseTrack, desiredNoiseRms);
        }

        // Onset truth is taken from the signal that actually reaches the microphone: after the
        // near-end RIR, after level normalization and after the scheduled delay. The dataset's
        // NearEndOnsetMs is only the insertion point of the clip - the speech inside it starts
        // later by however much leading silence the corpus recording carries, and that offset was
        // previously being charged to the product as detection latency.
        OracleActivity nearActivity = nearTrack.Length > 0
            ? OracleActivityAnalyzer.Analyze(
                nearTrack,
                scale: 1,
                SampleRate,
                startupGuardMs,
                minimumSpeechMs)
            : OracleActivity.Unavailable;

        return new RealtimeScene(
            echoImpulse,
            nearTrack,
            noiseTrack,
            onset,
            totalSamples,
            callbackSamples,
            testCase.ClockDriftPpm,
            testCase.FarEndNonlinear,
            speakerGain,
            nearActivity,
            nearActiveLevel,
            echoActiveTarget,
            echoReturnLossDb);
    }

    internal sealed class RealtimeScene
    {
        private readonly PartitionedFirFilter _echoFilter;
        private readonly float[] _nearTrack;
        private readonly float[] _noiseTrack;
        private readonly List<float> _renderHistory = [];
        private readonly double _clockRatio;
        private readonly bool _farEndNonlinear;
        private readonly double _speakerGain;
        private long _outputSamplePosition;
        private long _clippedSamples;
        private long _renderedSamples;

        public RealtimeScene(
            float[] echoImpulse,
            float[] nearTrack,
            float[] noiseTrack,
            int nearEndOnsetSample,
            int totalSamples,
            int callbackSamples,
            int clockDriftPpm,
            bool farEndNonlinear,
            double speakerGain,
            OracleActivity nearEndActivity,
            double nearEndActiveLevel,
            double echoActiveLevel,
            double echoReturnLossDb)
        {
            _echoFilter = new PartitionedFirFilter(echoImpulse, callbackSamples);
            _nearTrack = nearTrack;
            _noiseTrack = noiseTrack;
            NearEndOnsetSample = nearEndOnsetSample;
            TotalSamples = totalSamples;
            CallbackSamples = callbackSamples;
            _clockRatio = 1 + clockDriftPpm / 1_000_000.0;
            _farEndNonlinear = farEndNonlinear;
            _speakerGain = speakerGain;
            NearEndActivity = nearEndActivity;
            NearEndActiveLevel = nearEndActiveLevel;
            EchoActiveLevel = echoActiveLevel;
            EchoReturnLossDb = echoReturnLossDb;
        }

        /// <summary>
        /// Sample at which the near-end clip was inserted. This is a scheduling position, not an
        /// acoustic event: use <see cref="NearEndSpeechOnsetSample"/> or
        /// <see cref="NearEndEligibleOnsetSample"/> for anything latency related.
        /// </summary>
        public int NearEndOnsetSample { get; }

        public int TotalSamples { get; }
        public int CallbackSamples { get; }

        /// <summary>Activity truth measured on the delayed, reverberant, level-normalized near-end track.</summary>
        public OracleActivity NearEndActivity { get; }

        /// <summary>
        /// First sample at which near-end speech energy is actually present in the microphone
        /// signal. Firing before this is a genuine false trigger; -1 when there is no near-end.
        /// </summary>
        public long NearEndSpeechOnsetSample =>
            NearEndActivity.Available ? NearEndActivity.FirstSpeechSample : -1;

        /// <summary>
        /// Earliest sample at which the product is permitted to report a barge-in: the first point
        /// where enough continuous near-end speech exists to satisfy the minimum-duration rule and
        /// the playback startup guard has elapsed. This is the fair zero point for reaction latency,
        /// because no correct implementation may fire before it.
        /// </summary>
        public long NearEndEligibleOnsetSample =>
            NearEndActivity is { Available: true, Eligible: true }
                ? NearEndActivity.EligibleOnsetSample
                : -1;

        public double NearEndActiveLevel { get; }
        public double EchoActiveLevel { get; }
        public double EchoReturnLossDb { get; }
        public double SpeakerGain => _speakerGain;

        /// <summary>
        /// Fraction of microphone samples that hit the +/-0.98 clip. Reported so that a loud-echo
        /// case cannot silently become a clipping experiment.
        /// </summary>
        public double ClippedSampleRatio =>
            _renderedSamples > 0 ? _clippedSamples / (double)_renderedSamples : 0;

        public void Process(
            int offset,
            ReadOnlySpan<float> render,
            Span<float> microphone,
            Span<float> echo)
        {
            if (render.Length != CallbackSamples ||
                microphone.Length != CallbackSamples ||
                echo.Length != CallbackSamples)
            {
                throw new ArgumentException(
                    "Realtime scene buffers must match the configured callback size.");
            }

            for (int i = 0; i < render.Length; i++)
                _renderHistory.Add(render[i]);
            var speaker = new float[CallbackSamples];
            for (int i = 0; i < speaker.Length; i++)
            {
                double sourcePosition =
                    (_outputSamplePosition + i - DriftLookaheadSamples) * _clockRatio;
                int left = (int)Math.Floor(sourcePosition);
                float sample;
                if (left < 0 || left >= _renderHistory.Count)
                {
                    sample = 0;
                }
                else if (left + 1 >= _renderHistory.Count)
                {
                    sample = _renderHistory[left];
                }
                else
                {
                    double fraction = sourcePosition - left;
                    sample = (float)(
                        _renderHistory[left] +
                        (_renderHistory[left + 1] - _renderHistory[left]) * fraction);
                }
                // The loudspeaker drive level is a physical quantity derived from the echo target
                // and the echo return loss, so the nonlinearity below sees the level the speaker is
                // actually driven at rather than whatever level the TTS decoder emitted.
                speaker[i] = (float)(sample * _speakerGain);
            }
            if (_farEndNonlinear)
                ApplyLoudspeakerNonlinearity(speaker);
            _echoFilter.Process(speaker, echo);

            for (int i = 0; i < microphone.Length; i++)
            {
                int position = offset + i;
                float value = echo[i];
                if ((uint)position < (uint)_nearTrack.Length)
                    value += _nearTrack[position];
                if ((uint)position < (uint)_noiseTrack.Length)
                    value += _noiseTrack[position];
                float clamped = Math.Clamp(value, -0.98f, 0.98f);
                if (clamped != value)
                    _clippedSamples++;
                microphone[i] = clamped;
            }
            _renderedSamples += microphone.Length;
            _outputSamplePosition += CallbackSamples;
        }
    }

    internal sealed class PartitionedFirFilter
    {
        private readonly int _blockSize;
        private readonly int _fftSize;
        private readonly Complex[][] _impulsePartitions;
        private readonly Complex[][] _inputHistory;
        private readonly float[] _overlap;
        private int _historyIndex;

        public PartitionedFirFilter(float[] impulse, int blockSize)
        {
            _blockSize = blockSize;
            _fftSize = 1;
            while (_fftSize < checked(blockSize * 2))
                _fftSize = checked(_fftSize << 1);
            int partitionCount = Math.Max(
                1,
                (impulse.Length + blockSize - 1) / blockSize);
            _impulsePartitions = new Complex[partitionCount][];
            _inputHistory = new Complex[partitionCount][];
            _overlap = new float[blockSize];
            for (int partition = 0; partition < partitionCount; partition++)
            {
                var spectrum = new Complex[_fftSize];
                int sourceOffset = partition * blockSize;
                int count = Math.Min(blockSize, impulse.Length - sourceOffset);
                for (int i = 0; i < count; i++)
                    spectrum[i] = impulse[sourceOffset + i];
                Fft(spectrum, inverse: false);
                _impulsePartitions[partition] = spectrum;
                _inputHistory[partition] = new Complex[_fftSize];
            }
        }

        public void Process(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != _blockSize || output.Length != _blockSize)
                throw new ArgumentException("Partitioned FIR requires one complete block.");

            Complex[] current = _inputHistory[_historyIndex];
            Array.Clear(current);
            for (int i = 0; i < input.Length; i++)
                current[i] = input[i];
            Fft(current, inverse: false);

            var sum = new Complex[_fftSize];
            for (int partition = 0; partition < _impulsePartitions.Length; partition++)
            {
                int history = _historyIndex - partition;
                if (history < 0)
                    history += _inputHistory.Length;
                Complex[] inputSpectrum = _inputHistory[history];
                Complex[] impulseSpectrum = _impulsePartitions[partition];
                for (int bin = 0; bin < sum.Length; bin++)
                    sum[bin] += inputSpectrum[bin] * impulseSpectrum[bin];
            }
            Fft(sum, inverse: true);
            for (int i = 0; i < _blockSize; i++)
            {
                output[i] = (float)sum[i].Real + _overlap[i];
                _overlap[i] = (float)sum[i + _blockSize].Real;
            }
            _historyIndex++;
            if (_historyIndex == _inputHistory.Length)
                _historyIndex = 0;
        }
    }

    internal static float[] PrepareImpulse(float[] source)
    {
        if (source.Length == 0)
            return [1];

        int peakIndex = 0;
        float peak = 0;
        for (int i = 0; i < source.Length; i++)
        {
            float magnitude = Math.Abs(source[i]);
            if (magnitude > peak)
            {
                peak = magnitude;
                peakIndex = i;
            }
        }
        if (peak < 1e-8f)
            return [1];

        int start = Math.Max(0, peakIndex - 32);
        int length = Math.Min(MaximumRirSamples, source.Length - start);
        var impulse = new float[length];
        source.AsSpan(start, length).CopyTo(impulse);
        double mean = impulse.Average(x => (double)x);
        double energy = 0;
        for (int i = 0; i < impulse.Length; i++)
        {
            impulse[i] -= (float)mean;
            energy += impulse[i] * impulse[i];
        }
        double scale = energy > 1e-12 ? 1 / Math.Sqrt(energy) : 1;
        for (int i = 0; i < impulse.Length; i++)
            impulse[i] = (float)(impulse[i] * scale);
        return impulse;
    }

    internal static float[] Convolve(float[] signal, float[] impulse)
    {
        if (signal.Length == 0 || impulse.Length == 0)
            return [];
        if (impulse.Length == 1)
            return signal.Select(x => x * impulse[0]).ToArray();

        int resultLength = checked(signal.Length + impulse.Length - 1);
        int fftLength = 1;
        while (fftLength < resultLength)
            fftLength = checked(fftLength << 1);

        var left = new Complex[fftLength];
        var right = new Complex[fftLength];
        for (int i = 0; i < signal.Length; i++)
            left[i] = signal[i];
        for (int i = 0; i < impulse.Length; i++)
            right[i] = impulse[i];
        Fft(left, inverse: false);
        Fft(right, inverse: false);
        for (int i = 0; i < fftLength; i++)
            left[i] *= right[i];
        Fft(left, inverse: true);

        var result = new float[resultLength];
        for (int i = 0; i < result.Length; i++)
            result[i] = (float)left[i].Real;
        return result;
    }

    private static void Fft(Complex[] values, bool inverse)
    {
        int count = values.Length;
        for (int i = 1, j = 0; i < count; i++)
        {
            int bit = count >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                (values[i], values[j]) = (values[j], values[i]);
        }

        for (int length = 2; length <= count; length <<= 1)
        {
            double angle = 2 * Math.PI / length * (inverse ? 1 : -1);
            Complex step = new(Math.Cos(angle), Math.Sin(angle));
            for (int start = 0; start < count; start += length)
            {
                Complex phase = Complex.One;
                int half = length >> 1;
                for (int offset = 0; offset < half; offset++)
                {
                    Complex even = values[start + offset];
                    Complex odd = values[start + offset + half] * phase;
                    values[start + offset] = even + odd;
                    values[start + offset + half] = even - odd;
                    phase *= step;
                }
            }
        }
        if (!inverse)
            return;
        for (int i = 0; i < count; i++)
            values[i] /= count;
    }

    private static float[] ApplyClockDrift(float[] source, int ppm)
    {
        if (ppm == 0)
            return source.ToArray();
        double ratio = 1 + ppm / 1_000_000.0;
        var result = new float[source.Length];
        for (int i = 0; i < result.Length; i++)
        {
            double position = i * ratio;
            int left = (int)Math.Floor(position);
            if (left >= source.Length - 1)
                break;
            double fraction = position - left;
            result[i] = (float)(
                source[left] +
                (source[left + 1] - source[left]) * fraction);
        }
        return result;
    }

    private static void ApplyLoudspeakerNonlinearity(Span<float> samples)
    {
        const double drive = 1.8;
        double denominator = Math.Tanh(drive);
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)(Math.Tanh(samples[i] * drive) / denominator);
    }

    private static float[] Delay(float[] source, int samples)
    {
        if (samples <= 0)
            return source;
        var result = new float[checked(source.Length + samples)];
        source.CopyTo(result, samples);
        return result;
    }

    private static float[] Repeat(float[] source, int length)
    {
        var result = new float[length];
        for (int i = 0; i < result.Length; i++)
            result[i] = source[i % source.Length];
        return result;
    }

    private static void Add(Span<float> destination, ReadOnlySpan<float> source)
    {
        int count = Math.Min(destination.Length, source.Length);
        for (int i = 0; i < count; i++)
            destination[i] += source[i];
    }

    /// <summary>
    /// Normalizes a speech signal so its active speech level (RMS over speech frames only) hits the
    /// target. Unlike <see cref="ScaleToRms"/> this is invariant to how much silence the clip
    /// carries, so a fixed target produces a fixed perceived level across a mixed corpus.
    /// </summary>
    private static void ScaleToActiveSpeechLevel(Span<float> samples, double targetActiveLevel)
    {
        double level = OracleActivityAnalyzer.ActiveSpeechLevel(samples, SampleRate);
        if (level < 1e-9)
            return;
        double scale = targetActiveLevel / level;
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)(samples[i] * scale);
    }

    private static void ScaleToRms(Span<float> samples, double targetRms)
    {
        double rms = Rms(samples, 0, samples.Length);
        if (rms < 1e-9)
            return;
        double scale = targetRms / rms;
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)(samples[i] * scale);
    }

    private static double Rms(ReadOnlySpan<float> samples, int offset, int length)
    {
        if (length <= 0 || offset < 0 || offset >= samples.Length)
            return 0;
        int count = Math.Min(length, samples.Length - offset);
        double sum = 0;
        for (int i = 0; i < count; i++)
        {
            double value = samples[offset + i];
            sum += value * value;
        }
        return Math.Sqrt(sum / count);
    }
}
