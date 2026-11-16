namespace VoiceBox.DNR.DPDFNet.Internal
{
    /// <summary>
    /// Vorbis 窗 (亦称 KBD 近似 / sin² 窗)，DPDFNet 的 STFT/ISTFT 分析-合成窗。
    ///
    /// 定义 (与参考实现逐元素一致)：
    ///   s[n]  = sin(π/2 · (n + 0.5) / (N/2))
    ///   w[n]  = sin(π/2 · s[n]²)
    /// 该窗满足 Princen-Bradley 条件 w[n]² + w[n+N/2]² = 1，配合 50% 重叠-相加 (hop=N/2)
    /// 在恒等映射下可完美重建，因此分析与合成使用同一个窗。
    /// </summary>
    internal static class VorbisWindow
    {
        /// <summary>生成长度为 <paramref name="windowLength"/> 的 Vorbis 窗。</summary>
        public static float[] Create(int windowLength)
        {
            if (windowLength <= 0)
                throw new ArgumentOutOfRangeException(nameof(windowLength), windowLength, "windowLength must be > 0.");

            double halfSize = windowLength / 2.0;
            float[] window = new float[windowLength];
            for (int n = 0; n < windowLength; n++)
            {
                double s = Math.Sin(0.5 * Math.PI * (n + 0.5) / halfSize);
                window[n] = (float)Math.Sin(0.5 * Math.PI * s * s);
            }
            return window;
        }
    }
}
