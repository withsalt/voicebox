using System;
using VoiceBox.ASR.Qwen3.Internal;

namespace VoiceBox.ASR.Qwen3.Streaming
{
    /// <summary>
    /// Rolling re-transcription state machine. Mirrors the Python
    /// <c>ASRStreamingState</c> contract:
    ///   - First <c>UnfixedChunkNum</c> rounds run without a prefix prompt to let
    ///     the model converge.
    ///   - Subsequent rounds re-encode the cumulative waveform but seed the prompt
    ///     with the previously decoded text minus its last <c>UnfixedTokenNum</c>
    ///     BPE tokens, which damps boundary jitter near the latest acoustic frames.
    ///
    /// The planner is stateless: callers retain the running raw decoded string and
    /// feed it back in each round.
    /// </summary>
    internal static class StreamingPromptPlanner
    {
        public static string? BuildPrefix(
            int chunkId,
            string accumulatedRaw,
            int unfixedChunkNum,
            int unfixedTokenNum,
            Qwen3AsrTokenizer tokenizer)
        {
            if (chunkId < unfixedChunkNum) return null;
            if (string.IsNullOrEmpty(accumulatedRaw)) return null;
            if (unfixedTokenNum <= 0) return accumulatedRaw;

            int total = tokenizer.CountTokens(accumulatedRaw);
            int keep = total - unfixedTokenNum;
            if (keep <= 0) return string.Empty;

            // Binary-search the longest prefix substring whose token count is
            // ≤ keep. Each probe is one CountTokens call (~O(n)); total work is
            // O(log n · n) instead of the O(n²) shrink-by-char loop.
            int lo = 0;
            int hi = accumulatedRaw.Length;
            int best = 0;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1;
                int t = tokenizer.CountTokens(accumulatedRaw.AsSpan(0, mid).ToString());
                if (t <= keep)
                {
                    best = mid;
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }
            return accumulatedRaw.Substring(0, best);
        }
    }
}
