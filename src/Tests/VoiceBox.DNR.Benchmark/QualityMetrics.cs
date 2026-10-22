namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// 参考类信号域客观指标 (纯 C# 精确计算, 零外部依赖)。所有函数要求两路<b>已时间对齐</b>
    /// (见 <see cref="AudioAlign"/>); 内部只取公共长度。SI-SDR 天然尺度不变; 分段 SNR 先按最小二乘
    /// 全局增益把 estimate 对齐到 reference, 从而只度量频谱/时域失真而不受整体音量差影响。
    /// </summary>
    internal static class QualityMetrics
    {
        /// <summary>SI-SDR 在数值上的封顶 (dB), 避免 estimate≈reference 时除零得到 +∞。</summary>
        private const double SiSdrCeilingDb = 80.0;

        /// <summary>
        /// Scale-Invariant SDR (dB, 越高越好)。
        /// s_target = (&lt;e,s&gt;/&lt;s,s&gt;)·s; e_noise = e − s_target; SI-SDR = 10·log10(||s_target||²/||e_noise||²)。
        /// reference 全静音时无定义, 返回 NaN (由聚合层跳过)。
        /// </summary>
        public static double SiSdr(ReadOnlySpan<float> reference, ReadOnlySpan<float> estimate)
        {
            int n = Math.Min(reference.Length, estimate.Length);
            if (n == 0) return double.NaN;

            double dot = 0.0, refE = 0.0;
            for (int i = 0; i < n; i++)
            {
                dot += (double)reference[i] * estimate[i];
                refE += (double)reference[i] * reference[i];
            }
            if (refE <= 1e-12) return double.NaN; // 静音参考, SI-SDR 无定义

            double alpha = dot / refE;
            double targetE = 0.0, noiseE = 0.0;
            for (int i = 0; i < n; i++)
            {
                double t = alpha * reference[i];
                double err = estimate[i] - t;
                targetE += t * t;
                noiseE += err * err;
            }
            if (noiseE <= 1e-12) return SiSdrCeilingDb;       // 完美还原
            if (targetE <= 1e-12) return -SiSdrCeilingDb;     // estimate 与 reference 正交
            double db = 10.0 * Math.Log10(targetE / noiseE);
            return Math.Clamp(db, -SiSdrCeilingDb, SiSdrCeilingDb);
        }

        /// <summary>
        /// 分段 SNR (dB, 越高越好)。逐帧 (默认 25ms/10ms) 计 10·log10(sig/err), 每帧钳到 [-10,35] dB
        /// 后对<b>语音活跃帧</b>取均值 (静音帧按能量门限剔除, 避免 −10 dB 底噪拉低均值)。
        /// estimate 先做全局最小二乘增益对齐, 使指标不受整体音量差影响。无活跃帧时返回 NaN。
        /// </summary>
        public static double SegmentalSnr(ReadOnlySpan<float> reference, ReadOnlySpan<float> estimate,
            int frame = 400, int hop = 160)
        {
            int n = Math.Min(reference.Length, estimate.Length);
            if (n < frame) return double.NaN;

            // 全局最小二乘增益: alpha = <e,s>/<s,s>, 把 estimate 缩放到 reference 量级。
            double dot = 0.0, refE = 0.0;
            for (int i = 0; i < n; i++)
            {
                dot += (double)reference[i] * estimate[i];
                refE += (double)reference[i] * reference[i];
            }
            if (refE <= 1e-12) return double.NaN;
            double alpha = dot / refE;
            if (Math.Abs(alpha) < 1e-9) alpha = 1.0; // estimate 近全零, 不缩放, 让误差如实反映

            // 语音活跃门限: 整段每样本均方能量的 1e-3 (≈ -30 dB), 低于此的帧视为静音跳过。
            double meanSq = refE / n;
            double activeFloor = meanSq * 1e-3;

            double sum = 0.0;
            int count = 0;
            for (int start = 0; start + frame <= n; start += hop)
            {
                double sig = 0.0, err = 0.0;
                for (int j = 0; j < frame; j++)
                {
                    double r = reference[start + j];
                    double e = alpha * estimate[start + j];
                    sig += r * r;
                    double d = r - e;
                    err += d * d;
                }
                if (sig / frame < activeFloor) continue; // 静音帧剔除
                double snr = 10.0 * Math.Log10((sig + 1e-12) / (err + 1e-12));
                sum += Math.Clamp(snr, -10.0, 35.0);
                count++;
            }
            return count == 0 ? double.NaN : sum / count;
        }
    }
}
