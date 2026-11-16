using VoiceBox.ASR.FireRedASR2.Internal.Utils;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Maps a token sequence to per-token start/end times via CTC forced alignment over
    /// frame-level log-probabilities (flat <c>[frameCount, vocabSize]</c> layout).
    /// </summary>
    internal static class CtcTimestampAligner
    {
        /// <summary>
        /// Fallback encoder frame shift (seconds) used when the caller does not supply one.
        /// Corresponds to 10 ms fbank shift × 4 encoder subsampling.
        /// </summary>
        public const double DefaultFrameShiftSeconds = 0.04;

        public static List<int[]> Align(
            float[] logProbs,
            int frameCount,
            int vocabSize,
            List<int> tokens,
            int blankId,
            double frameShiftSeconds = DefaultFrameShiftSeconds,
            CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            if (tokens is null || tokens.Count == 0) return new List<int[]>();
            if (tokens.Count > frameCount) return new List<int[]>();

            int[] alignment;
            try
            {
                alignment = ComputeHelper.ForcedAlign(logProbs, frameCount, vocabSize, tokens.ToArray(), blankId, ct);
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
                return new List<int[]>();
            }

            var starts = new List<double>(tokens.Count);
            var ends = new List<double>(tokens.Count);
            int prevToken = blankId;

            for (int t = 0; t < alignment.Length; t++)
            {
                ct.ThrowIfCancellationRequested();
                int token = alignment[t];
                if (token != blankId)
                {
                    if (token != prevToken)
                    {
                        if (prevToken != blankId) ends.Add(t * frameShiftSeconds);
                        starts.Add(t * frameShiftSeconds);
                        prevToken = token;
                    }
                }
                else if (prevToken != blankId)
                {
                    ends.Add(t * frameShiftSeconds);
                    prevToken = blankId;
                }
            }
            if (prevToken != blankId) ends.Add(alignment.Length * frameShiftSeconds);

            if (starts.Count != tokens.Count) return new List<int[]>();
            return ToMilliseconds(starts, ends);
        }

        private static List<int[]> ToMilliseconds(List<double> starts, List<double> ends)
        {
            var result = new List<int[]>(starts.Count);
            for (int i = 0; i < starts.Count; i++)
            {
                result.Add(new[]
                {
                    (int)Math.Round(starts[i] * 1000, MidpointRounding.AwayFromZero),
                    (int)Math.Round(ends[i] * 1000, MidpointRounding.AwayFromZero),
                });
            }
            return result;
        }
    }
}
