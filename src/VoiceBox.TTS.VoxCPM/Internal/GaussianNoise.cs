namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// 确定性高斯噪声源, 作为 CFM 扩散的起点 (每 patch 一个 [feat_dim, patch] 块)。
    /// 生产环境用 (seed) 保证可复现; 它与 numpy 的 default_rng 实现不同, 因此生成的音频是
    /// "另一份同样自然"的实现, 不会与 Python golden 逐样本相同 —— 这是生成式模型的正常行为。
    /// 一致性测试通过加载导出的精确噪声 (noise_*.npy) 来逐样本对齐 C# 的数学正确性。
    /// 基于 SplitMix64 + Box-Muller, 数值稳定且与平台无关。
    /// </summary>
    internal sealed class GaussianNoise
    {
        private ulong _state;
        private double _spare;
        private bool _hasSpare;

        public GaussianNoise(long seed) => _state = unchecked((ulong)seed) + 0x9E3779B97F4A7C15UL;

        private double NextUniform()
        {
            // SplitMix64
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            return (z >> 11) * (1.0 / 9007199254740992.0);
        }

        public double NextGaussian()
        {
            if (_hasSpare) { _hasSpare = false; return _spare; }
            double u1, u2;
            do { u1 = NextUniform(); } while (u1 <= 1e-12);
            u2 = NextUniform();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            _spare = mag * Math.Sin(2.0 * Math.PI * u2);
            _hasSpare = true;
            return mag * Math.Cos(2.0 * Math.PI * u2);
        }

        public float[] Fill(int count)
        {
            var a = new float[count];
            for (int i = 0; i < count; i++) a[i] = (float)NextGaussian();
            return a;
        }
    }
}
