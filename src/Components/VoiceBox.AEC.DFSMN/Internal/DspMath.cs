namespace VoiceBox.AEC.DFSMN.Internal
{
    internal static class DspMath
    {
        public static float[] HammingWindow(int length)
        {
            if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));
            if (length == 1) return [1f];

            var window = new float[length];
            double denom = length - 1;
            for (int i = 0; i < length; i++)
                window[i] = (float)(0.54 - 0.46 * Math.Cos(2.0 * Math.PI * i / denom));
            return window;
        }

        public static int NextPowerOfTwo(int value)
        {
            if (value <= 0) return 1;
            int v = value - 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            return v + 1;
        }

        /// <summary>
        /// 衰减 dB → 线性增益 (10^(-db/20))。掩码下限/残差抑制/近端保护共用同一转换,
        /// 保证"保护未接合时与旧行为逐位一致"的等价性不因各处手写表达式漂移而破裂。
        /// </summary>
        public static float AttenuationDbToGain(float attenuationDb)
            => MathF.Pow(10f, -attenuationDb / 20f);

        /// <summary>一阶平滑系数: exp(-hopMs / timeConstantMs), 时间常数下限 1e-3ms。</summary>
        public static float OnePoleCoefficient(float hopMs, float timeConstantMs)
            => MathF.Exp(-hopMs / MathF.Max(1e-3f, timeConstantMs));

        public static float Clamp(float value, float min, float max)
            => value < min ? min : value > max ? max : value;

        public static float MelScale(float freq) => 1127f * MathF.Log(1f + freq / 700f);
    }
}
