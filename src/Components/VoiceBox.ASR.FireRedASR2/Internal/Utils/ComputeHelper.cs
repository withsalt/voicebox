namespace VoiceBox.ASR.FireRedASR2.Internal.Utils
{
    internal static class ComputeHelper
    {
        /// <summary>
        /// CTC forced alignment: given per-frame log probabilities and a target token
        /// sequence, return the per-frame alignment maximising the path probability.
        /// <paramref name="logProbs"/> is flat <c>[frameCount * vocabSize]</c>; the dp and
        /// backtrack tables are also flat, and backtrack stores only the 0/1/2 delta from
        /// the current state to the chosen predecessor (one byte per cell vs the legacy
        /// <see cref="int"/> two-d array).
        /// </summary>
        public static int[] ForcedAlign(float[] logProbs, int frameCount, int vocabSize, int[] targets, int blank,
            CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            int t = frameCount;
            if (t == 0) return Array.Empty<int>();
            int s = targets.Length;

            int minFrames = 2 * s + 1;
            if (t < minFrames)
                throw new ArgumentException($"Not enough frames: need at least {minFrames}, got {t}.");

            // Expanded target: blank, t1, blank, t2, ..., blank
            int l = 2 * s + 1;
            int[] extTargets = new int[l];
            for (int i = 0; i < l; i++)
                extTargets[i] = (i % 2 == 0) ? blank : targets[i / 2];

            // Flat dp/backtrack: index by frame * l + state.
            float[] dp = new float[(long)t * l <= int.MaxValue ? t * l : throw new InvalidOperationException("ForcedAlign DP size overflow.")];
            // backtrack delta: 0 = same state, 1 = state-1, 2 = state-2, 255 = invalid.
            byte[] backtrack = new byte[t * l];

            for (int i = 0; i < l; i++) dp[i] = float.NegativeInfinity;
            dp[0] = logProbs[blank];
            backtrack[0] = 255;
            if (l > 1)
            {
                dp[1] = logProbs[targets[0]];
                backtrack[1] = 255;
            }

            for (int frame = 1; frame < t; frame++)
            {
                ct.ThrowIfCancellationRequested();
                int rowOffset = frame * vocabSize;
                int dpRow = frame * l;
                int dpPrev = (frame - 1) * l;
                for (int state = 0; state < l; state++)
                {
                    int currentLabel = extTargets[state];

                    float best = dp[dpPrev + state];
                    int bestDelta = 0;

                    if (state > 0)
                    {
                        float v = dp[dpPrev + state - 1];
                        if (v > best) { best = v; bestDelta = 1; }
                    }
                    if (state > 1 && extTargets[state] != blank && extTargets[state - 2] != extTargets[state])
                    {
                        float v = dp[dpPrev + state - 2];
                        if (v > best) { best = v; bestDelta = 2; }
                    }

                    if (best == float.NegativeInfinity)
                    {
                        dp[dpRow + state] = float.NegativeInfinity;
                        backtrack[dpRow + state] = 255;
                    }
                    else
                    {
                        dp[dpRow + state] = best + logProbs[rowOffset + currentLabel];
                        backtrack[dpRow + state] = (byte)bestDelta;
                    }
                }
            }

            int lastRow = (t - 1) * l;
            int finalPos = l > 1 ? (dp[lastRow + l - 1] > dp[lastRow + l - 2] ? l - 1 : l - 2) : 0;
            if (dp[lastRow + finalPos] == float.NegativeInfinity)
                throw new InvalidOperationException("Failed to find a valid CTC alignment path.");

            int[] alignment = new int[t];
            int pos = finalPos;
            for (int frame = t - 1; frame >= 0; frame--)
            {
                ct.ThrowIfCancellationRequested();
                alignment[frame] = extTargets[pos];
                if (frame > 0)
                {
                    byte delta = backtrack[frame * l + pos];
                    if (delta == 255)
                        throw new InvalidOperationException("Invalid pointer encountered while backtracking.");
                    pos -= delta;
                }
            }

            // Sanity check: decoded sequence (after blank/dup removal) must match the targets.
            int prevToken = blank;
            int decodedCount = 0;
            for (int i = 0; i < alignment.Length; i++)
            {
                ct.ThrowIfCancellationRequested();
                int token = alignment[i];
                if (token != blank)
                {
                    if (token != prevToken)
                    {
                        decodedCount++;
                        prevToken = token;
                    }
                }
                else prevToken = blank;
            }
            if (decodedCount != s)
                throw new InvalidOperationException($"CTC alignment decoded {decodedCount} tokens but target has {s}.");

            return alignment;
        }
    }
}
