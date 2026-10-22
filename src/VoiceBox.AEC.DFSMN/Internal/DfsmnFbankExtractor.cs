namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class DfsmnFbankExtractor
    {
        private const int SampleRate = 16000;
        private const float LowFreqHz = 20f;
        private const float LogFloor = 1.1920929e-7f;
        private const float Preemphasis = 0.97f;

        private readonly int _frameLength;
        private readonly int _fftSize;
        private readonly int _numBins;
        private readonly float[] _window;
        private readonly MelFilter[] _melFilters;
        private readonly Radix2Fft _fft;
        private readonly float[] _fftRe;
        private readonly float[] _fftIm;

        public DfsmnFbankExtractor(int frameLength, int numBins)
        {
            if (frameLength <= 0) throw new ArgumentOutOfRangeException(nameof(frameLength));
            if (numBins <= 0) throw new ArgumentOutOfRangeException(nameof(numBins));

            _frameLength = frameLength;
            _fftSize = DspMath.NextPowerOfTwo(frameLength);
            _numBins = numBins;
            _window = DspMath.HammingWindow(frameLength);
            _melFilters = BuildMelFilters(numBins, _fftSize, SampleRate, LowFreqHz, SampleRate / 2f);
            _fft = new Radix2Fft(_fftSize);
            _fftRe = new float[_fftSize];
            _fftIm = new float[_fftSize];
        }

        /// <summary>
        /// Compute one Kaldi-style 80-bin log fbank frame from int16-scale samples.
        /// </summary>
        public void ComputeFrame(ReadOnlySpan<float> frameInt16Scale, Span<float> destination)
        {
            if (frameInt16Scale.Length != _frameLength)
                throw new ArgumentException($"frame length must be {_frameLength}.", nameof(frameInt16Scale));
            if (destination.Length != _numBins)
                throw new ArgumentException($"destination length must be {_numBins}.", nameof(destination));

            float mean = 0f;
            for (int i = 0; i < _frameLength; i++) mean += frameInt16Scale[i];
            mean /= _frameLength;

            for (int i = 0; i < _frameLength; i++)
                _fftRe[i] = frameInt16Scale[i] - mean;

            for (int i = _frameLength - 1; i >= 1; i--)
                _fftRe[i] -= Preemphasis * _fftRe[i - 1];
            _fftRe[0] -= Preemphasis * _fftRe[0];

            for (int i = 0; i < _frameLength; i++)
                _fftRe[i] *= _window[i];
            Array.Clear(_fftRe, _frameLength, _fftSize - _frameLength);
            Array.Clear(_fftIm);

            _fft.Forward(_fftRe, _fftIm);

            int numFftBins = _fftSize / 2; // Kaldi fbank excludes Nyquist for mel filters.
            for (int b = 0; b < _numBins; b++)
            {
                MelFilter filter = _melFilters[b];
                ReadOnlySpan<float> weights = filter.Weights;
                double sum = 0.0;
                for (int i = 0; i < weights.Length; i++)
                {
                    int k = filter.StartBin + i;
                    float w = weights[i];
                    float re = _fftRe[k];
                    float im = _fftIm[k];
                    sum += w * (re * re + im * im);
                }

                float v = (float)sum;
                destination[b] = MathF.Log(v < LogFloor ? LogFloor : v);
            }
        }

        private static MelFilter[] BuildMelFilters(int numBins, int fftSize, int sampleRate, float lowFreq, float highFreq)
        {
            int numFftBins = fftSize / 2;
            float fftBinWidth = sampleRate / (float)fftSize;
            float melLow = DspMath.MelScale(lowFreq);
            float melHigh = DspMath.MelScale(highFreq);
            float melDelta = (melHigh - melLow) / (numBins + 1);

            var filters = new MelFilter[numBins];
            for (int bin = 0; bin < numBins; bin++)
            {
                float left = melLow + bin * melDelta;
                float center = melLow + (bin + 1) * melDelta;
                float right = melLow + (bin + 2) * melDelta;
                var weights = new List<float>();
                int startBin = -1;
                for (int k = 0; k < numFftBins; k++)
                {
                    float mel = DspMath.MelScale(fftBinWidth * k);
                    if (mel <= left || mel >= right) continue;
                    startBin = startBin < 0 ? k : startBin;
                    weights.Add(mel <= center
                        ? (mel - left) / (center - left)
                        : (right - mel) / (right - center));
                }
                filters[bin] = new MelFilter(
                    startBin < 0 ? 0 : startBin,
                    weights.ToArray());
            }
            return filters;
        }

        private readonly record struct MelFilter(int StartBin, float[] Weights);
    }
}
