using System.Numerics.Tensors;

namespace VoiceBox.AEC.DFSMN.Internal
{
    /// <summary>
    /// Sample-by-sample normalized least-mean-squares (NLMS) adaptive echo canceller.
    ///
    /// This is the same NLMS recursion as a textbook circular-buffer implementation. The
    /// per-sample filter output, reference energy and coefficient update use contiguous spans.
    ///
    /// Layout:
    /// <list type="bullet">
    /// <item><description><c>_history</c> is a length-<c>2N</c> buffer holding the most recent
    /// <c>N</c> reference samples as a single contiguous, chronological window
    /// <c>_history[_head .. _head+N)</c> (oldest first, newest last). When the window reaches
    /// the end of the buffer it is re-packed to the front.</description></item>
    /// <item><description><c>_weights</c> stores the filter coefficients in the SAME order as the
    /// window (the coefficient for the newest tap last), so the convolution is a plain dot
    /// product and the update is a plain multiply-add over aligned spans.</description></item>
    /// </list>
    ///
    /// </summary>
    internal sealed class AdaptiveNlmsEchoCanceller
    {
        private readonly float[] _weights;
        private readonly float[] _history;
        private readonly int _filterLength;
        private readonly float _mu;
        private readonly float _epsilon;
        private readonly float _leakage;
        private int _head;

        public AdaptiveNlmsEchoCanceller(int filterLength, float adaptationStep, float epsilon, float leakage)
        {
            if (filterLength <= 0) throw new ArgumentOutOfRangeException(nameof(filterLength));
            if (adaptationStep < 0 || float.IsNaN(adaptationStep)) throw new ArgumentOutOfRangeException(nameof(adaptationStep));
            if (epsilon <= 0 || float.IsNaN(epsilon)) throw new ArgumentOutOfRangeException(nameof(epsilon));
            if (leakage <= 0 || leakage > 1 || float.IsNaN(leakage)) throw new ArgumentOutOfRangeException(nameof(leakage));

            _filterLength = filterLength;
            _weights = new float[filterLength];
            _history = new float[2 * filterLength];
            _mu = adaptationStep;
            _epsilon = epsilon;
            _leakage = leakage;
        }

        public void Process(
            ReadOnlySpan<float> capture,
            ReadOnlySpan<float> render,
            Span<float> linear,
            Span<float> echo,
            bool adapt = true)
        {
            if (render.Length != capture.Length || linear.Length != capture.Length || echo.Length != capture.Length)
                throw new ArgumentException("all NLMS buffers must have equal length.");

            int n = _filterLength;
            float[] weights = _weights;
            float[] history = _history;
            int head = _head;

            for (int i = 0; i < capture.Length; i++)
            {
                // Slide the window forward by one and write the newest reference sample at its tail.
                if (head + n == history.Length)
                {
                    // Window reached the end of the backing buffer: keep the newest N-1 samples,
                    // drop the oldest, and re-pack to the front so the window stays contiguous.
                    Array.Copy(history, head + 1, history, 0, n - 1);
                    head = 0;
                    history[n - 1] = render[i];
                }
                else
                {
                    history[head + n] = render[i];
                    head++;
                }

                ReadOnlySpan<float> window = history.AsSpan(head, n);

                float estimate = TensorPrimitives.Dot<float>(weights, window);
                float norm = TensorPrimitives.SumOfSquares<float>(window) + _epsilon;

                float e = capture[i] - estimate;
                echo[i] = estimate;
                linear[i] = e;

                // Freeze the coefficients during double-talk: still slide the reference window and
                // emit the echo/linear estimates above, but skip the weight update so near-end
                // speech in the error term cannot drag the converged filter off the echo path.
                if (!adapt || _mu <= 0f || norm <= _epsilon) continue;
                float scale = _mu * e / norm;
                // weights[k] += scale * window[k]
                TensorPrimitives.MultiplyAdd<float>(window, scale, weights, weights);
            }

            _head = head;

            // Leakage decays the coefficients toward zero; skip it while frozen so a long
            // interruption does not slowly erode the converged filter and let echo back in.
            if (adapt && _leakage < 1f)
                TensorPrimitives.Multiply<float>(weights, _leakage, weights);
        }

        public void Reset()
        {
            Array.Clear(_weights);
            Array.Clear(_history);
            _head = 0;
        }
    }
}
