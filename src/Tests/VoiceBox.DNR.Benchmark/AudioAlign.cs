namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// 时间对齐工具。DNR 流水线 (流式 STFT/ISTFT、重采样、FIFO 预填) 给输出引入延迟。SI-SDR / 分段 SNR
    /// 对时移<b>极其敏感</b> (实测差 27 样本即可让 +13 dB 崩到 -10 dB), 且神经降噪对干净/含噪输入的
    /// 有效延迟略有不同, 故必须<b>逐对</b>对齐、且直接对准 SI-SDR 峰而非仅相关峰:
    ///   1. 每后端预热时用「干净→引擎」估一个粗粒度标称滞后 (<see cref="EstimateNominalLag"/>);
    ///   2. 每个评分对在标称附近细化: 先归一化互相关取峰, 再在其邻域内取<b>使 SI-SDR 最大</b>的整数滞后
    ///      (<see cref="AlignBestLag"/>), 抵消内容相关的抖动与分数延迟残差。
    /// 延迟本身作为「算法延迟」单列上报, 不与降噪质量混淆。
    /// </summary>
    internal static class AudioAlign
    {
        /// <summary>
        /// 粗粒度标称滞后: 归一化互相关在 ±<paramref name="maxLag"/> 内取峰。返回正数表示 estimate 落后
        /// reference 若干样本 (estimate[i+lag] ≈ reference[i])。每后端一次, 作逐对细化的搜索中心。
        /// </summary>
        public static int EstimateNominalLag(ReadOnlySpan<float> reference, ReadOnlySpan<float> estimate,
            int maxLag, int window = 24000)
        {
            if (!SafeWindow(reference.Length, estimate.Length, PeakIndex(reference), window, maxLag,
                    out int start, out int w))
                return 0;
            return ArgmaxNormXcorr(reference, estimate, start, w, -maxLag, maxLag);
        }

        /// <summary>
        /// 逐对细化滞后: 在 [center-radius, center+radius] 内取<b>使 SI-SDR 最大</b>的整数滞后 (直接对准评分指标,
        /// 而非仅相关峰 —— 二者对 HPF/相移类处理可差十余样本, 而 SI-SDR 对此又极敏感)。窗取 reference 能量峰段,
        /// 全局搜索。<paramref name="reference"/>=clean, <paramref name="estimate"/>=enhanced。
        /// </summary>
        public static int AlignBestLag(ReadOnlySpan<float> reference, ReadOnlySpan<float> estimate,
            int center, int radius, int window = 16000)
        {
            int lo = center - radius, hi = center + radius;
            int margin = Math.Max(Math.Abs(lo), Math.Abs(hi));
            if (!SafeWindow(reference.Length, estimate.Length, PeakIndex(reference), window, margin,
                    out int start, out int w))
                return center;

            int bestLag = center;
            double bestScore = double.NegativeInfinity;
            for (int d = lo; d <= hi; d++)
            {
                double s = WindowSiSdr(reference, estimate, start, w, d);
                if (s > bestScore) { bestScore = s; bestLag = d; }
            }
            return bestLag;
        }

        /// <summary>
        /// 按 <paramref name="lag"/> 把两路裁到公共区间: 返回等长的 (referenceAligned, estimateAligned),
        /// 满足 estimateAligned[i] ≈ referenceAligned[i]。lag&gt;0 丢 estimate 前导; lag&lt;0 丢 reference 前导。
        /// </summary>
        public static (float[] refAligned, float[] estAligned) TrimToAligned(
            ReadOnlySpan<float> reference, ReadOnlySpan<float> estimate, int lag)
        {
            int refStart = lag < 0 ? -lag : 0;
            int estStart = lag > 0 ? lag : 0;
            int n = Math.Min(reference.Length - refStart, estimate.Length - estStart);
            if (n <= 0) return (Array.Empty<float>(), Array.Empty<float>());
            return (reference.Slice(refStart, n).ToArray(), estimate.Slice(estStart, n).ToArray());
        }

        /// <summary>reference 短时能量峰所在样本下标 (hop=160), 作相关窗中心以抗静音干扰。</summary>
        private static int PeakIndex(ReadOnlySpan<float> reference)
        {
            const int hop = 160;
            int peak = 0;
            double best = -1.0;
            for (int i = 0; i + hop <= reference.Length; i += hop)
            {
                double e = 0.0;
                for (int j = 0; j < hop; j++) e += (double)reference[i + j] * reference[i + j];
                if (e > best) { best = e; peak = i; }
            }
            return peak;
        }

        /// <summary>
        /// 选一个两侧各留 <paramref name="margin"/> 余量的相关窗, 保证所有 ±margin 的位移访问都在界内。
        /// 信号过短 (不足 2·margin) 时返回 false。
        /// </summary>
        private static bool SafeWindow(int refLen, int estLen, int peak, int window, int margin,
            out int start, out int w)
        {
            start = 0; w = 0;
            int limit = Math.Min(refLen, estLen);
            w = Math.Min(window, limit - 2 * margin);
            if (w <= 0) return false;
            start = Math.Clamp(peak - w / 2, margin, limit - margin - w);
            return true;
        }

        /// <summary>归一化互相关在 [lo,hi] 内取峰 (reference 窗能量对各 lag 恒定, 故只需按 estimate 能量归一)。</summary>
        private static int ArgmaxNormXcorr(ReadOnlySpan<float> reference, ReadOnlySpan<float> estimate,
            int start, int w, int lo, int hi)
        {
            int bestLag = 0;
            double bestScore = double.NegativeInfinity;
            for (int d = lo; d <= hi; d++)
            {
                double dot = 0.0, estE = 0.0;
                for (int i = 0; i < w; i++)
                {
                    float e = estimate[start + i + d];
                    dot += (double)reference[start + i] * e;
                    estE += (double)e * e;
                }
                double score = dot / Math.Sqrt(estE + 1e-12);
                if (score > bestScore) { bestScore = score; bestLag = d; }
            }
            return bestLag;
        }

        /// <summary>在相关窗上按位移 d 算 SI-SDR (dB), 供细化取峰; reference[start+i] vs estimate[start+i+d]。</summary>
        private static double WindowSiSdr(ReadOnlySpan<float> reference, ReadOnlySpan<float> estimate,
            int start, int w, int d)
        {
            double dot = 0.0, refE = 0.0;
            for (int i = 0; i < w; i++)
            {
                dot += (double)reference[start + i] * estimate[start + i + d];
                refE += (double)reference[start + i] * reference[start + i];
            }
            if (refE <= 1e-12) return double.NegativeInfinity;
            double alpha = dot / refE;
            double targetE = 0.0, noiseE = 0.0;
            for (int i = 0; i < w; i++)
            {
                double t = alpha * reference[start + i];
                double err = estimate[start + i + d] - t;
                targetE += t * t;
                noiseE += err * err;
            }
            if (noiseE <= 1e-12) return double.PositiveInfinity;
            return 10.0 * Math.Log10((targetE + 1e-12) / (noiseE + 1e-12));
        }
    }
}
