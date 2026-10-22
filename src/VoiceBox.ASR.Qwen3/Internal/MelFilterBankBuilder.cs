using System;

namespace VoiceBox.ASR.Qwen3.Internal
{
    /// <summary>
    /// Builds the Whisper-style log-mel filterbank that HuggingFace's
    /// <c>WhisperFeatureExtractor</c> uses when the preprocessor config does not
    /// ship precomputed filters.
    /// </summary>
    /// <remarks>
    /// Layout matches <c>transformers.audio_utils.mel_filter_bank</c> with
    /// <c>norm="slaney"</c> and <c>mel_scale="slaney"</c>:
    ///   * Linear mel below 1 kHz (slope 3/200), log mel above.
    ///   * Slaney area-normalisation per triangular filter so equal-energy
    ///     bandwidth gets equal weight regardless of bin width.
    /// Output shape: <c>[num_mel_filters][num_frequency_bins]</c> row-major.
    /// </remarks>
    internal static class MelFilterBankBuilder
    {
        public static float[][] Build(
            int numFrequencyBins,
            int numMelFilters,
            double minFrequency,
            double maxFrequency,
            int samplingRate)
        {
            if (numFrequencyBins <= 1) throw new ArgumentOutOfRangeException(nameof(numFrequencyBins));
            if (numMelFilters <= 0) throw new ArgumentOutOfRangeException(nameof(numMelFilters));
            if (maxFrequency <= minFrequency) throw new ArgumentException("maxFrequency must be greater than minFrequency.");

            double minMel = HertzToMel(minFrequency);
            double maxMel = HertzToMel(maxFrequency);
            double melStep = (maxMel - minMel) / (numMelFilters + 1);

            // Edge frequencies of each triangular filter on the linear-Hz axis.
            double[] edgeFreqs = new double[numMelFilters + 2];
            for (int i = 0; i < edgeFreqs.Length; i++)
                edgeFreqs[i] = MelToHertz(minMel + i * melStep);

            // FFT bin centre frequencies.
            double[] fftFreqs = new double[numFrequencyBins];
            double nyquist = samplingRate / 2.0;
            for (int i = 0; i < numFrequencyBins; i++)
                fftFreqs[i] = i * nyquist / (numFrequencyBins - 1);

            var filters = new float[numMelFilters][];
            for (int m = 0; m < numMelFilters; m++)
            {
                double left = edgeFreqs[m];
                double centre = edgeFreqs[m + 1];
                double right = edgeFreqs[m + 2];
                double leftWidth = centre - left;
                double rightWidth = right - centre;

                // Slaney normalization: 2 / (mel_freqs[m+2] - mel_freqs[m])
                double enorm = 2.0 / (right - left);

                var filter = new float[numFrequencyBins];
                for (int k = 0; k < numFrequencyBins; k++)
                {
                    double f = fftFreqs[k];
                    double weight;
                    if (f <= left || f >= right) weight = 0.0;
                    else if (f <= centre) weight = (f - left) / leftWidth;
                    else weight = (right - f) / rightWidth;
                    filter[k] = (float)(weight * enorm);
                }
                filters[m] = filter;
            }
            return filters;
        }

        // Slaney-style mel scale: linear below 1 kHz, logarithmic above.
        private const double SlaneyMinLogHz = 1000.0;
        private static readonly double SlaneyMinLogMel = SlaneyMinLogHz / (200.0 / 3.0); // = 15
        private static readonly double SlaneyLogStep = Math.Log(6.4) / 27.0;

        private static double HertzToMel(double freq)
        {
            if (freq < SlaneyMinLogHz)
                return freq / (200.0 / 3.0);
            return SlaneyMinLogMel + Math.Log(freq / SlaneyMinLogHz) / SlaneyLogStep;
        }

        private static double MelToHertz(double mel)
        {
            if (mel < SlaneyMinLogMel)
                return mel * (200.0 / 3.0);
            return SlaneyMinLogHz * Math.Exp(SlaneyLogStep * (mel - SlaneyMinLogMel));
        }
    }
}
