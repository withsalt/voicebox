namespace VoiceBox.DNR.Hush.Internal
{
    /// <summary>
    /// 深度滤波 (Deep Filtering) 算子，端口自 deepfilter-rt 的 <c>apply_deep_filter_flat</c>。
    ///
    /// 对前 <c>nbDf</c> 个频点，用网络输出的 df_order 阶复数系数，对最近 df_order 帧的
    /// <b>带噪</b>复数谱做因果复数 FIR：
    ///   out[f] = Σ_{o=0..order-1} x[o][f] · coef[f][o]   (复数乘)
    /// 其中 x[0] 为最旧帧、x[order-1] 为当前帧 (df_lookahead=0)。
    /// </summary>
    internal static class DeepFilterOp
    {
        /// <summary>
        /// 施加深度滤波，结果写回 <paramref name="outRe"/> / <paramref name="outIm"/> 的前 <paramref name="nbDf"/> 个频点。
        /// </summary>
        /// <param name="outRe">输出帧实部 (长度 ≥ freqSize)，前 nbDf 个被覆盖。</param>
        /// <param name="outIm">输出帧虚部 (长度 ≥ freqSize)，前 nbDf 个被覆盖。</param>
        /// <param name="xRe">滚动带噪谱实部，按帧拼接 (帧 o 位于 [o*freqSize..])，oldest-first，共 dfOrder 帧。</param>
        /// <param name="xIm">滚动带噪谱虚部，布局同上。</param>
        /// <param name="coefs">系数，布局 [freq][order*2]，每对 (re, im)，长度 nbDf*dfOrder*2。</param>
        /// <param name="nbDf">深度滤波频点数。</param>
        /// <param name="dfOrder">滤波阶数 (帧数)。</param>
        /// <param name="freqSize">每帧谱的频点数 (fft/2+1)。</param>
        public static void Apply(
            Span<float> outRe, Span<float> outIm,
            ReadOnlySpan<float> xRe, ReadOnlySpan<float> xIm,
            ReadOnlySpan<float> coefs,
            int nbDf, int dfOrder, int freqSize)
        {
            for (int freq = 0; freq < nbDf; freq++)
            {
                float re = 0f;
                float im = 0f;
                int coefBase = freq * dfOrder * 2;
                for (int order = 0; order < dfOrder; order++)
                {
                    int sIdx = order * freqSize + freq;
                    float sRe = xRe[sIdx];
                    float sIm = xIm[sIdx];
                    float cr = coefs[coefBase + order * 2];
                    float ci = coefs[coefBase + order * 2 + 1];
                    // (sRe + sIm·i) · (cr + ci·i)
                    re += sRe * cr - sIm * ci;
                    im += sRe * ci + sIm * cr;
                }
                outRe[freq] = re;
                outIm[freq] = im;
            }
        }
    }
}
