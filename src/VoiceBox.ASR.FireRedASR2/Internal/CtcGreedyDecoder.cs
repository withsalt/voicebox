namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Pure greedy CTC decoder over the flat <c>[frameCount, vocabSize]</c> logits layout
    /// produced by <see cref="AedInferenceEngine"/>. Picks per-frame argmax (first-wins on
    /// ties, matching PyTorch / NumPy), then collapses duplicates and blanks.
    /// </summary>
    internal static class CtcGreedyDecoder
    {
        public static List<int> Decode(float[] ctcLogits, int frameCount, int vocabSize, int blankId = 0,
            CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            int t = frameCount;
            if (t == 0 || vocabSize == 0) return new List<int>();

            int[] argmax = new int[t];
            for (int j = 0; j < t; j++)
            {
                ct.ThrowIfCancellationRequested();
                int rowOffset = j * vocabSize;
                int best = 0;
                float bestVal = ctcLogits[rowOffset];
                for (int k = 1; k < vocabSize; k++)
                {
                    float v = ctcLogits[rowOffset + k];
                    if (v > bestVal)
                    {
                        bestVal = v;
                        best = k;
                    }
                }
                argmax[j] = best;
            }

            return RemoveDuplicatesAndBlanks(argmax, blankId, ct);
        }

        public static List<int> RemoveDuplicatesAndBlanks(int[] sequence, int blankId, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            if (sequence.Length == 0) return new List<int>();

            var decoded = new List<int>(sequence.Length);
            int prev = -1;
            foreach (int token in sequence)
            {
                ct.ThrowIfCancellationRequested();
                if (token != prev && token != blankId) decoded.Add(token);
                prev = token;
            }
            return decoded;
        }
    }
}
