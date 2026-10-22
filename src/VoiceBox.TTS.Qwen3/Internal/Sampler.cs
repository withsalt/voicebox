namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Token sampling matching HF generate's warper order: temperature → top-k → top-p → softmax →
    /// multinomial. Greedy (do_sample=false) is argmax over the processed logits.
    /// </summary>
    internal static class Sampler
    {
        public static int Argmax(ReadOnlySpan<float> logits)
        {
            int best = 0;
            float bv = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
                if (logits[i] > bv) { bv = logits[i]; best = i; }
            return best;
        }

        public static int Sample(float[] logits, bool doSample, float temperature, int topK, float topP, Random rng)
        {
            if (!doSample) return Argmax(logits);

            int n = logits.Length;
            var work = (float[])logits.Clone();
            if (temperature > 0f && Math.Abs(temperature - 1f) > 1e-6f)
                for (int i = 0; i < n; i++) work[i] /= temperature;

            // top-k: keep only the k highest logits
            if (topK > 0 && topK < n)
            {
                float kth = KthLargest(work, topK);
                for (int i = 0; i < n; i++) if (work[i] < kth) work[i] = float.NegativeInfinity;
            }

            // softmax
            float max = float.NegativeInfinity;
            for (int i = 0; i < n; i++) if (work[i] > max) max = work[i];
            double sum = 0;
            var probs = new double[n];
            for (int i = 0; i < n; i++)
            {
                double e = float.IsNegativeInfinity(work[i]) ? 0.0 : Math.Exp(work[i] - max);
                probs[i] = e; sum += e;
            }
            if (sum <= 0) return Argmax(logits);
            for (int i = 0; i < n; i++) probs[i] /= sum;

            // top-p (nucleus): zero out the tail beyond cumulative p
            if (topP > 0f && topP < 1f)
            {
                int[] order = SortedDescIndices(probs);
                double cum = 0;
                bool cut = false;
                foreach (int idx in order)
                {
                    if (cut) { probs[idx] = 0; continue; }
                    cum += probs[idx];
                    if (cum >= topP) cut = true;   // keep this one, drop the rest
                }
                double s2 = 0;
                for (int i = 0; i < n; i++) s2 += probs[i];
                if (s2 <= 0) return order[0];
                for (int i = 0; i < n; i++) probs[i] /= s2;
            }

            // multinomial
            double r = rng.NextDouble();
            double acc = 0;
            for (int i = 0; i < n; i++)
            {
                acc += probs[i];
                if (r < acc) return i;
            }
            return Argmax(logits);
        }

        private static float KthLargest(float[] v, int k)
        {
            var copy = (float[])v.Clone();
            Array.Sort(copy);                     // ascending
            return copy[copy.Length - k];
        }

        private static int[] SortedDescIndices(double[] p)
        {
            var idx = new int[p.Length];
            for (int i = 0; i < idx.Length; i++) idx[i] = i;
            Array.Sort(idx, (a, b) => p[b].CompareTo(p[a]));
            return idx;
        }
    }
}
