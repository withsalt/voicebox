using System;
using System.Buffers;

namespace VoiceBox.ASR.Confucius4.Internal
{
    /// <summary>
    /// Whisper-style log-mel feature extractor matching <c>WhisperFeatureExtractor</c>
    /// from HuggingFace Transformers. Produces a (n_mels, T) array in row-major
    /// order (<c>data[m * frames + t]</c>) as expected by the Confucius4 ONNX encoder.
    /// </summary>
    /// <remarks>
    /// Pipeline (per Whisper paper / HF impl):
    ///   1. Reflect-pad waveform by n_fft/2 on both sides.
    ///   2. STFT with periodic Hann window (sym=false), hop=160, n_fft=400.
    ///   3. Magnitude squared → power spectrum.
    ///   4. Multiply by mel filterbank loaded from preprocessor_config.json.
    ///   5. log10, clamp at max - 8, then (x + 4) / 4.
    /// </remarks>
    internal sealed class WhisperFeatureExtractor
    {
        private readonly PreprocessorConfig _config;
        private readonly float[] _window;
        private readonly float[][] _melFilters;
        private readonly int[] _melStarts;
        private readonly int[] _melEnds;
        private readonly RealFft _fft;

        public int FeatureSize => _config.FeatureSize;
        public int HopLength => _config.HopLength;
        public int SamplingRate => _config.SamplingRate;
        public int NFft => _config.NFft;
        public int FftBins => _config.FftBins;

        public WhisperFeatureExtractor(PreprocessorConfig config)
        {
            ArgumentNullException.ThrowIfNull(config);

            _config = config;
            _window = BuildHannWindow(config.NFft);
            _melFilters = config.MelFilters is { Length: > 0 }
                ? config.MelFilters!
                : MelFilterBankBuilder.Build(
                    numFrequencyBins: config.FftBins,
                    numMelFilters: config.FeatureSize,
                    minFrequency: 0.0,
                    maxFrequency: config.SamplingRate / 2.0,
                    samplingRate: config.SamplingRate);
            _melStarts = new int[_melFilters.Length];
            _melEnds = new int[_melFilters.Length];
            for (int m = 0; m < _melFilters.Length; m++)
            {
                float[] filter = _melFilters[m];
                int start = 0;
                while (start < filter.Length && filter[start] == 0f) start++;
                int end = filter.Length;
                while (end > start && filter[end - 1] == 0f) end--;
                _melStarts[m] = start;
                _melEnds[m] = end;
            }
            _fft = new RealFft(config.NFft);
        }

        /// <summary>
        /// Extract a log-mel spectrogram for the input waveform. Returns an array in
        /// <c>(mels, frames)</c> row-major layout.
        /// </summary>
        public LogMelSpectrogram Compute(ReadOnlySpan<float> waveform)
        {
            int nFft = _config.NFft;
            int hop = _config.HopLength;
            int pad = nFft / 2;

            if (waveform.Length <= pad)
                return new LogMelSpectrogram(Array.Empty<float>(), FeatureSize, 0);

            int paddedLen = waveform.Length + 2 * pad;
            float[] padded = ArrayPool<float>.Shared.Rent(paddedLen);
            try
            {
                ReflectPad(waveform, padded.AsSpan(0, paddedLen), pad);

                // Whisper discards the final centered STFT frame (magnitudes[..., :-1]).
                int numFrames = (paddedLen - nFft) / hop;
                if (numFrames <= 0)
                    return new LogMelSpectrogram(Array.Empty<float>(), FeatureSize, 0);

                int nMels = _melFilters.Length;

                float[] mel = new float[nMels * numFrames];
                ComputeFramesIntoLog10(padded.AsSpan(0, paddedLen), mel, 0, numFrames, nMels, numFrames);
                NormalizeInPlace(mel.AsSpan(0, nMels * numFrames));
                return new LogMelSpectrogram(mel, nMels, numFrames);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(padded);
            }
        }

        /// <summary>
        /// Incremental log-mel computation for a monotonically-growing waveform.
        /// Only the new frames at the tail are re-FFT'd; the previously-computed
        /// log10 magnitudes are kept verbatim. The global max-floor normalisation
        /// is re-applied to the full buffer (O(N), no FFT) since the new tail can
        /// shift the global maximum.
        /// </summary>
        /// <remarks>
        /// Pre-normalisation log10 values are stored in <paramref name="state"/> across
        /// calls so we never re-FFT the head. The returned <see cref="LogMelSpectrogram"/>
        /// holds a freshly-normalised copy (mutating the cache would break the next
        /// incremental call).
        /// </remarks>
        public LogMelSpectrogram ComputeIncremental(ReadOnlySpan<float> waveform, IncrementalMelState state)
        {
            ArgumentNullException.ThrowIfNull(state);

            int nFft = _config.NFft;
            int hop = _config.HopLength;
            int pad = nFft / 2;
            int nMels = _melFilters.Length;

            if (waveform.Length <= pad)
                return new LogMelSpectrogram(Array.Empty<float>(), nMels, 0);

            int paddedLen = waveform.Length + 2 * pad;
            int numFrames = paddedLen >= nFft ? (paddedLen - nFft) / hop : 0;
            if (numFrames <= 0)
                return new LogMelSpectrogram(Array.Empty<float>(), nMels, 0);

            // Decide how many leading frames can be reused. Frame t reads
            // padded[t*hop .. t*hop+nFft); padded layout is
            //   [pad reflect-left | waveform | pad reflect-right].
            // To reuse frame t, every sample it reads must come from the previous
            // call's left-pad or waveform region (i.e. NOT from the trailing
            // reflect-pad that depended on the prior tail). That requires
            //   t*hop + nFft <= pad + prevSamples
            //   ⇒ t <= (pad + prevSamples - nFft) / hop.
            // Additionally the left reflect-pad must be unchanged; it reads
            // waveform[1..pad], so we also require prevSamples >= pad. With
            // pad=200 and a 16 kHz mic this means ≥12.5 ms of audio, trivially
            // true after the first chunk.
            int prevSamples = state.LastWaveformSamples;
            int reusable;
            if (prevSamples < pad || waveform.Length < pad)
            {
                reusable = 0;
            }
            else
            {
                int prevPaddedLen = prevSamples + 2 * pad;
                int prevFrames = prevPaddedLen >= nFft ? 1 + (prevPaddedLen - nFft) / hop : 0;
                int cleanFramesPrev = pad + prevSamples >= nFft
                    ? (pad + prevSamples - nFft) / hop + 1
                    : 0;
                reusable = Math.Min(prevFrames, cleanFramesPrev);
                reusable = Math.Min(reusable, numFrames);
            }

            float[] log10 = state.EnsureCapacity(nMels, numFrames, reusable);
            int stride = state.Stride;

            int startFrame = reusable;
            int recomputeFrames = numFrames - startFrame;
            if (recomputeFrames > 0)
            {
                float[] padded = ArrayPool<float>.Shared.Rent(paddedLen);
                try
                {
                    ReflectPad(waveform, padded.AsSpan(0, paddedLen), pad);
                    ComputeFramesIntoLog10(padded.AsSpan(0, paddedLen), log10, startFrame, recomputeFrames, nMels, stride);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(padded);
                }
            }

            state.Commit(numFrames, waveform.Length);

            // Repack the (possibly strided) cache into a tight (nMels, numFrames) buffer
            // for downstream consumers, then normalise the copy.
            float[] normalized = new float[nMels * numFrames];
            if (stride == numFrames)
            {
                log10.AsSpan(0, nMels * numFrames).CopyTo(normalized);
            }
            else
            {
                for (int m = 0; m < nMels; m++)
                    Array.Copy(log10, m * stride, normalized, m * numFrames, numFrames);
            }
            NormalizeInPlace(normalized.AsSpan(0, nMels * numFrames));
            return new LogMelSpectrogram(normalized, nMels, numFrames);
        }

        /// <summary>
        /// Compute log10(power*mel) for <paramref name="frameCount"/> frames starting at
        /// <paramref name="startFrame"/> and write them in row-major (mels, frames) layout
        /// (frames stride = <paramref name="stride"/>) into <paramref name="logOut"/>.
        /// Normalisation is NOT applied — callers do that across the full buffer.
        /// </summary>
        private void ComputeFramesIntoLog10(
            ReadOnlySpan<float> padded,
            float[] logOut, int startFrame, int frameCount, int nMels, int stride)
        {
            int nFft = _config.NFft;
            int hop = _config.HopLength;
            int fftBins = _config.FftBins;

            float[] frame = ArrayPool<float>.Shared.Rent(nFft);
            float[] power = ArrayPool<float>.Shared.Rent(fftBins);
            try
            {
                for (int idx = 0; idx < frameCount; idx++)
                {
                    int t = startFrame + idx;
                    int start = t * hop;
                    for (int i = 0; i < nFft; i++)
                        frame[i] = padded[start + i] * _window[i];

                    _fft.ForwardPower(frame.AsSpan(0, nFft), power.AsSpan(0, fftBins));

                    for (int m = 0; m < nMels; m++)
                    {
                        ReadOnlySpan<float> filter = _melFilters[m];
                        float sum = 0f;
                        int startBin = _melStarts[m];
                        int endBin = Math.Min(_melEnds[m], fftBins);
                        for (int k = startBin; k < endBin; k++)
                            sum += filter[k] * power[k];
                        logOut[m * stride + t] = MathF.Log10(MathF.Max(sum, 1e-10f));
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(frame);
                ArrayPool<float>.Shared.Return(power);
            }
        }

        private static void NormalizeInPlace(Span<float> mel)
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < mel.Length; i++)
            {
                float v = mel[i];
                if (v > max) max = v;
            }
            float floor = max - 8f;
            for (int i = 0; i < mel.Length; i++)
            {
                float v = mel[i];
                if (v < floor) v = floor;
                mel[i] = (v + 4f) * 0.25f;
            }
        }

        private static void ReflectPad(ReadOnlySpan<float> src, Span<float> dst, int pad)
        {
            for (int i = 0; i < pad; i++)
                dst[i] = src[pad - i];
            src.CopyTo(dst.Slice(pad, src.Length));
            int srcEnd = src.Length;
            for (int i = 0; i < pad; i++)
                dst[pad + srcEnd + i] = src[srcEnd - 2 - i];
        }

        private static float[] BuildHannWindow(int n)
        {
            var window = new float[n];
            for (int i = 0; i < n; i++)
                window[i] = 0.5f * (1f - MathF.Cos(2f * MathF.PI * i / n));
            return window;
        }
    }

    /// <summary>Compact log-mel spectrogram with row-major <c>(mels, frames)</c> layout.</summary>
    internal readonly struct LogMelSpectrogram
    {
        public float[] Data { get; }
        public int NumMels { get; }
        public int NumFrames { get; }

        public LogMelSpectrogram(float[] data, int numMels, int numFrames)
        {
            Data = data;
            NumMels = numMels;
            NumFrames = numFrames;
        }
    }

    /// <summary>
    /// Persistent across-call buffer for <see cref="WhisperFeatureExtractor.ComputeIncremental"/>.
    /// Holds pre-normalisation log10(mel·power) values in <c>(nMels, capacityFrames)</c>
    /// row-major layout — the row stride is the buffer capacity, NOT the current
    /// frame count, so growth never has to repack the head columns. Thread affinity
    /// matches the Confucius4 streaming pump (single-writer per session); no internal locking.
    /// </summary>
    internal sealed class IncrementalMelState
    {
        private float[] _log10 = Array.Empty<float>();
        private int _capacityFrames;
        private int _nMels;

        public int LastNumFrames { get; private set; }
        public int LastWaveformSamples { get; private set; }

        /// <summary>Row stride of the cache buffer (≥ <see cref="LastNumFrames"/>).</summary>
        public int Stride => _capacityFrames;

        /// <summary>
        /// Ensure the cache can hold <paramref name="numFrames"/> columns, preserving
        /// the first <paramref name="keepFrames"/> columns from the previous call.
        /// Returns the internal buffer; new columns at [keepFrames, numFrames) are
        /// uninitialised and must be filled by the caller before use.
        /// </summary>
        public float[] EnsureCapacity(int nMels, int numFrames, int keepFrames)
        {
            if (nMels <= 0) throw new ArgumentOutOfRangeException(nameof(nMels));
            if (numFrames <= 0) throw new ArgumentOutOfRangeException(nameof(numFrames));
            if (keepFrames < 0 || keepFrames > numFrames)
                throw new ArgumentOutOfRangeException(nameof(keepFrames));
            if (_nMels != 0 && _nMels != nMels)
                throw new InvalidOperationException("nMels changed across incremental calls.");

            _nMels = nMels;

            if (_capacityFrames >= numFrames)
                return _log10;

            int newCapacity = _capacityFrames == 0
                ? Math.Max(numFrames, 1024)
                : Math.Max(numFrames, _capacityFrames * 2);

            var bigger = new float[nMels * newCapacity];
            int preserve = Math.Min(keepFrames, LastNumFrames);
            if (preserve > 0)
            {
                for (int m = 0; m < nMels; m++)
                {
                    Array.Copy(
                        _log10, m * _capacityFrames,
                        bigger, m * newCapacity,
                        preserve);
                }
            }
            _log10 = bigger;
            _capacityFrames = newCapacity;
            return _log10;
        }

        public void Commit(int numFrames, int waveformSamples)
        {
            LastNumFrames = numFrames;
            LastWaveformSamples = waveformSamples;
        }

        public void Reset()
        {
            LastNumFrames = 0;
            LastWaveformSamples = 0;
        }
    }
}
