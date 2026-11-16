using System.Buffers;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using VoiceBox.ASR.FireRedASR2.Internal.Model;
using VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Streaming front-end: turns waveform samples into CMVN-normalised log-mel features.
    /// </summary>
    internal sealed class WavFrontend : IDisposable
    {
        private const float WaveformScale = 32768f;
        private const double VarianceFloor = 1e-20;

        private readonly FrontendConfEntity _config;
        private readonly OnlineFbank _onlineFbank;
        private readonly CmvnEntity _cmvn;
        private bool _disposed;

        public WavFrontend(string mvnFilePath, FrontendConfEntity config)
        {
            ArgumentNullException.ThrowIfNull(config);

            _config = config;
            _onlineFbank = new OnlineFbank(
                dither: config.Dither,
                snip_edges: config.SnipEdges,
                sample_rate: config.SampleRate,
                num_bins: config.NumMels,
                window_type: config.Window);
            _cmvn = LoadCmvn(mvnFilePath);
        }

        public float[] GetFbank(float[] samples)
        {
            ArgumentNullException.ThrowIfNull(samples);
            return GetFbank(samples.AsSpan());
        }

        /// <summary>
        /// Compute FBank features for <paramref name="samples"/>. Internally rents a
        /// scratch buffer from <see cref="ArrayPool{T}"/> to scale into the 16-bit PCM
        /// range expected by <see cref="OnlineFbank"/> without allocating on every call.
        /// </summary>
        public float[] GetFbank(ReadOnlySpan<float> samples)
        {
            if (samples.IsEmpty) return Array.Empty<float>();

            float[] rented = ArrayPool<float>.Shared.Rent(samples.Length);
            try
            {
                // Scale into 16-bit PCM range in place; the legacy `samples.AsEnumerable().Select(...)` chain
                // allocated a new array each frame.
                for (int i = 0; i < samples.Length; i++)
                    rented[i] = samples[i] * WaveformScale;
                return _onlineFbank.GetFbank(rented, samples.Length);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        public void InputFinished() => _onlineFbank.InputFinished();

        /// <summary>
        /// Apply (x - mean) * invStddev per frame, in place.
        /// </summary>
        public float[] ApplyCmvn(float[] inputs)
        {
            ArgumentNullException.ThrowIfNull(inputs);

            float[] negMean = _cmvn.NegativeMeans;
            float[] invStd = _cmvn.InverseStddevs;
            int dim = negMean.Length;
            int frames = inputs.Length / dim;

            for (int frame = 0; frame < frames; frame++)
            {
                int baseIndex = frame * dim;
                ApplyRow(inputs.AsSpan(baseIndex, dim), negMean, invStd);
            }
            return inputs;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ApplyRow(Span<float> row, ReadOnlySpan<float> negMean, ReadOnlySpan<float> invStd)
        {
            int k = 0;
            if (Vector.IsHardwareAccelerated)
            {
                int width = Vector<float>.Count;
                for (; k <= row.Length - width; k += width)
                {
                    var v = new Vector<float>(row.Slice(k, width));
                    var m = new Vector<float>(negMean.Slice(k, width));
                    var s = new Vector<float>(invStd.Slice(k, width));
                    ((v - m) * s).CopyTo(row.Slice(k, width));
                }
            }
            for (; k < row.Length; k++)
                row[k] = (row[k] - negMean[k]) * invStd[k];
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _onlineFbank.Dispose();
        }

        private static CmvnEntity LoadCmvn(string mvnFilePath)
        {
            if (string.IsNullOrEmpty(mvnFilePath))
                throw new ArgumentException("mvn file path must be provided.", nameof(mvnFilePath));

            List<double> means = new();
            List<double> vars = new();
            int section = 0; // 0 = none, 1 = AddShift, 2 = Rescale

            using var reader = new StreamReader(mvnFilePath);
            while (reader.ReadLine() is { } line)
            {
                if (line.StartsWith("<AddShift>", StringComparison.Ordinal)) { section = 1; continue; }
                if (line.StartsWith("<Rescale>", StringComparison.Ordinal)) { section = 2; continue; }
                if (!line.StartsWith("<LearnRateCoef>", StringComparison.Ordinal)) continue;
                if (section is not 1 and not 2) continue;

                List<double> target = section == 1 ? means : vars;
                ParseBracketedDoubles(line, target);
            }

            if (means.Count == 0 || vars.Count == 0)
                throw new InvalidDataException($"Invalid CMVN file: {mvnFilePath}");

            // Trailing count column: divide by count, drop it from means; convert vars to inverse stddev.
            double count = means[^1];
            int dim = means.Count - 1;

            float[] negMean = new float[dim];
            float[] invStd = new float[dim];

            for (int i = 0; i < dim; i++)
            {
                double m = means[i] / count;
                double v = vars[i] / count - m * m;
                if (v < VarianceFloor) v = VarianceFloor;
                negMean[i] = (float)m;
                invStd[i] = (float)(1.0 / Math.Sqrt(v));
            }

            return new CmvnEntity(negMean, invStd);
        }

        private static void ParseBracketedDoubles(string line, List<double> output)
        {
            int open = line.IndexOf('[');
            int close = line.LastIndexOf(']');
            if (open < 0 || close <= open) return;

            string body = line.Substring(open + 1, close - open - 1);
            foreach (string token in body.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                output.Add(double.Parse(token.Trim(), CultureInfo.InvariantCulture));
        }
    }
}
