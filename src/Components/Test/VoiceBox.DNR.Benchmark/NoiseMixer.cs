using System.Text;

namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// 给干净 utterance 叠加确定性白噪声到指定 SNR, 作为无外部噪声源时的回退。
    /// 噪声按 <paramref name="stableKey"/> (通常含路径+SNR) 的 FNV-1a 稳定哈希播种, 因此:
    ///   - 同一 (utterance, SNR) 永远得到同一段噪声 (跨后端、跨进程一致), 保证横评公平;
    ///   - 不依赖 String.GetHashCode (后者每进程随机化, 不可复现)。
    /// SNR 以整段功率计 (含首尾静音); 叠加后钳到 [-1,1] 模拟采集削顶。
    /// 复制自 VoiceBox.ASR.Qwen3.Benchmark.NoiseMixer。
    /// </summary>
    internal static class NoiseMixer
    {
        /// <summary>返回 clean + 缩放白噪声 (新数组)。<paramref name="snrDb"/> 越小噪声越强。</summary>
        public static float[] AddWhiteNoise(float[] clean, double snrDb, string stableKey, int globalSeed)
        {
            ArgumentNullException.ThrowIfNull(clean);
            if (clean.Length == 0) return clean;

            double signalPower = 0.0;
            for (int i = 0; i < clean.Length; i++) signalPower += (double)clean[i] * clean[i];
            signalPower /= clean.Length;
            if (signalPower <= 0.0) return (float[])clean.Clone(); // 全静音, 无可定义 SNR

            int seed = unchecked((int)StableHash(stableKey) ^ globalSeed);
            var rng = new Random(seed);

            var noise = new float[clean.Length];
            double noisePower = 0.0;
            for (int i = 0; i < clean.Length; i++)
            {
                float g = NextGaussian(rng);
                noise[i] = g;
                noisePower += (double)g * g;
            }
            noisePower /= clean.Length;
            if (noisePower <= 0.0) return (float[])clean.Clone();

            double targetNoisePower = signalPower / Math.Pow(10.0, snrDb / 10.0);
            float scale = (float)Math.Sqrt(targetNoisePower / noisePower);

            var outp = new float[clean.Length];
            for (int i = 0; i < clean.Length; i++)
            {
                float v = clean[i] + scale * noise[i];
                outp[i] = v > 1f ? 1f : (v < -1f ? -1f : v);
            }
            return outp;
        }

        /// <summary>Box-Muller 标准正态。</summary>
        private static float NextGaussian(Random rng)
        {
            double u1 = 1.0 - rng.NextDouble(); // (0,1], 避免 log(0)
            double u2 = 1.0 - rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        /// <summary>FNV-1a 32-bit, 跨进程稳定。</summary>
        internal static uint StableHash(string s)
        {
            const uint offset = 2166136261;
            const uint prime = 16777619;
            uint hash = offset;
            foreach (byte b in Encoding.UTF8.GetBytes(s))
            {
                hash ^= b;
                hash *= prime;
            }
            return hash;
        }
    }
}
