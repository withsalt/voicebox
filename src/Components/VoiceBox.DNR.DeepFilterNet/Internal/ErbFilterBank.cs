namespace VoiceBox.DNR.DeepFilterNet.Internal
{
    /// <summary>
    /// ERB (Equivalent Rectangular Bandwidth) 滤波器组带宽表，端口自 libDF <c>erb_fb()</c>。
    ///
    /// 返回每个 ERB 带覆盖的 FFT 频点数 (band widths)，长度 = nbBands，总和 = fft/2+1。
    /// 该数组同时用于：能量降维 (compute_band_corr) 与增益升维 (apply_interp_band_gain)。
    /// </summary>
    internal static class ErbFilterBank
    {
        /// <summary>Hz → ERB 刻度：9.265 · ln(1 + f/(24.7·9.265))。</summary>
        public static float Freq2Erb(float freqHz) => 9.265f * MathF.Log(1f + freqHz / (24.7f * 9.265f));

        /// <summary>ERB 刻度 → Hz：24.7·9.265·(exp(e/9.265) − 1)。</summary>
        public static float Erb2Freq(float nErb) => 24.7f * 9.265f * (MathF.Exp(nErb / 9.265f) - 1f);

        /// <summary>
        /// 构造 ERB 带宽表 (每带的频点数)。与 libDF 逐元素一致。
        /// </summary>
        /// <param name="sr">采样率 (Hz)。</param>
        /// <param name="fftSize">FFT 大小。</param>
        /// <param name="nbBands">ERB 带数 (nb_erb)。</param>
        /// <param name="minNbFreqs">每带最少频点数 (min_nb_erb_freqs)。</param>
        public static int[] Compute(int sr, int fftSize, int nbBands, int minNbFreqs)
        {
            int nyqFreq = sr / 2;
            float freqWidth = (float)sr / fftSize;
            float erbLow = Freq2Erb(0f);
            float erbHigh = Freq2Erb(nyqFreq);
            int[] erb = new int[nbBands];
            float step = (erbHigh - erbLow) / nbBands;

            int prevFreq = 0;   // 上一个 ERB 带的最后频点
            int freqOver = 0;   // 已计入前面带的"借用"频点数
            for (int i = 1; i <= nbBands; i++)
            {
                float f = Erb2Freq(erbLow + i * step);
                // Rust f32::round 为四舍五入 (远离零)，须用 AwayFromZero 与之一致。
                int fb = (int)MathF.Round(f / freqWidth, MidpointRounding.AwayFromZero);
                int nbFreqs = fb - prevFreq - freqOver;
                if (nbFreqs < minNbFreqs)
                {
                    freqOver = minNbFreqs - nbFreqs; // 记录强制补足的频点数
                    nbFreqs = minNbFreqs;            // 强制最少 minNbFreqs
                }
                else
                {
                    freqOver = 0;
                }
                erb[i - 1] = nbFreqs;
                prevFreq = fb;
            }

            erb[nbBands - 1] += 1; // 因为共有 fft/2+1 个频点

            int sum = 0;
            for (int i = 0; i < nbBands; i++) sum += erb[i];
            int tooLarge = sum - (fftSize / 2 + 1);
            if (tooLarge > 0) erb[nbBands - 1] -= tooLarge;

            return erb;
        }
    }
}
