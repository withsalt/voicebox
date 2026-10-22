using System.Runtime.InteropServices;
using Concentus;

namespace VoiceBox.Audio.Common.Resampling
{
    /// <summary>
    /// Single-channel streaming float32 resampler backed by Concentus' SpeexDSP resampler.
    /// The wrapper keeps VoiceBox's old Process/Flush/Reset shape while centralizing
    /// Concentus' frame-count and tail-drain semantics.
    /// </summary>
    public sealed class StreamingSpeexResampler : IDisposable
    {
        private readonly int _inputRate;
        private readonly int _outputRate;
        private readonly int _quality;
        private readonly IResampler _resampler;

        private float[] _zeroPad = [];
        private bool _started;
        private bool _flushed;
        private bool _disposed;

        public int InputRate => _inputRate;

        public int OutputRate => _outputRate;

        public int Quality => _quality;

        public TimeSpan OutputLatency => _resampler.OutputLatency;

        public int OutputLatencySamples => _resampler.OutputLatencySamples;

        public StreamingSpeexResampler(int inputRate, int outputRate, int quality = 10)
        {
            if (inputRate <= 0) throw new ArgumentOutOfRangeException(nameof(inputRate));
            if (outputRate <= 0) throw new ArgumentOutOfRangeException(nameof(outputRate));
            if (quality is < 0 or > 10) throw new ArgumentOutOfRangeException(nameof(quality), "Quality must be between 0 and 10.");

            _inputRate = inputRate;
            _outputRate = outputRate;
            _quality = quality;
            _resampler = ResamplerFactory.CreateResampler(numChannels: 1, inputRate, outputRate, quality);
            Reset();
        }

        public int EstimateOutputLength(int inputLength)
        {
            if (inputLength <= 0) return 0;

            double scaled = Math.Ceiling(inputLength * (double)_outputRate / _inputRate);
            double withMargin = scaled + _resampler.OutputLatencySamples + 16;
            return withMargin >= int.MaxValue ? int.MaxValue : (int)withMargin;
        }

        public int Process(ReadOnlySpan<float> input, ref float[] output)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (input.IsEmpty) return 0;

            _started = true;
            _flushed = false;
            return ProcessCore(input, ref output);
        }

        public int Flush(ref float[] output)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_started || _flushed) return 0;

            int pad = Math.Max(1, _resampler.InputLatency + 2);
            if (_zeroPad.Length < pad)
                _zeroPad = new float[pad];
            else
                Array.Clear(_zeroPad, 0, pad);

            int produced = ProcessCore(_zeroPad.AsSpan(0, pad), ref output);
            _flushed = true;
            return produced;
        }

        public void Reset()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _resampler.ResetMem();
            _resampler.SkipZeroes();
            _started = false;
            _flushed = false;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _resampler.Dispose();
            _zeroPad = [];
            _started = false;
            _flushed = true;
            _disposed = true;
        }

        private int ProcessCore(ReadOnlySpan<float> input, ref float[] output)
        {
            int producedTotal = 0;
            int consumedTotal = 0;
            EnsureCapacity(ref output, EstimateOutputLength(input.Length));

            while (consumedTotal < input.Length)
            {
                if (producedTotal >= output.Length)
                    EnsureCapacity(ref output, Math.Max(output.Length + 64, output.Length * 2));

                ReadOnlySpan<float> remaining = input[consumedTotal..];
                int inLen = remaining.Length;
                int outLen = output.Length - producedTotal;

                _resampler.Process(
                    channel_index: 0,
                    AsWritableSpan(remaining),
                    ref inLen,
                    output.AsSpan(producedTotal),
                    ref outLen);

                if (inLen < 0 || outLen < 0)
                    throw new InvalidOperationException("Concentus resampler returned a negative consumed or produced count.");

                consumedTotal += inLen;
                producedTotal += outLen;

                if (inLen == 0 && outLen == 0)
                {
                    if (producedTotal >= output.Length)
                    {
                        EnsureCapacity(ref output, Math.Max(output.Length + 64, output.Length * 2));
                        continue;
                    }

                    throw new InvalidOperationException("Concentus resampler made no progress while input remained.");
                }
            }

            return producedTotal;
        }

        private static Span<float> AsWritableSpan(ReadOnlySpan<float> input)
            // Concentus 2.2.2 requires Span<float> but only reads input in Process.
            => MemoryMarshal.CreateSpan(ref MemoryMarshal.GetReference(input), input.Length);

        private static void EnsureCapacity(ref float[] buffer, int needed)
        {
            if (needed <= buffer.Length) return;

            int newSize = buffer.Length == 0 ? Math.Max(needed, 256) : buffer.Length * 2;
            while (newSize < needed)
                newSize *= 2;
            buffer = new float[newSize];
        }
    }
}
