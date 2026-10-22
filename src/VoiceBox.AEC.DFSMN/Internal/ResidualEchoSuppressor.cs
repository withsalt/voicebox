namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class ResidualEchoSuppressor
    {
        private readonly float _minGain;
        private readonly float _ratio;
        private readonly float _attack;
        private readonly float _release;
        private float _gain = 1f;

        public ResidualEchoSuppressor(
            float maxSuppressionDb,
            float ratio,
            float attackMs,
            float releaseMs,
            int hopLength,
            int sampleRate)
        {
            _minGain = maxSuppressionDb <= 0 ? 1f : DspMath.AttenuationDbToGain(maxSuppressionDb);
            _ratio = ratio;
            float hopMs = 1000f * hopLength / sampleRate;
            _attack = DspMath.OnePoleCoefficient(hopMs, attackMs);
            _release = DspMath.OnePoleCoefficient(hopMs, releaseMs);
        }

        public void Process(ReadOnlySpan<float> echoEstimate, Span<float> output)
            => Process(echoEstimate, output, gainFloor: 0f);

        /// <summary>
        /// <paramref name="gainFloor"/> 限定本 hop 的最小增益 (0 = 不限制, 旧行为)。
        /// 近端保护在双讲帧把该下限抬向 1: 标量抑制是全带宽的, 双讲时它削的正是
        /// 从掩码下幸存的用户语音。下限抬升立即生效 (不等慢释放); 叠说结束后增益被
        /// 保护包络的释放曲线托着缓降 (与掩码地板同步), 包络归零后才按 attack
        /// 重新进入全额抑制。
        /// </summary>
        public void Process(ReadOnlySpan<float> echoEstimate, Span<float> output, float gainFloor)
        {
            if (echoEstimate.Length != output.Length)
                throw new ArgumentException("echoEstimate and output must have the same length.");

            double echoEnergy = 1e-12;
            double outEnergy = 1e-12;
            for (int i = 0; i < output.Length; i++)
            {
                echoEnergy += echoEstimate[i] * echoEstimate[i];
                outEnergy += output[i] * output[i];
            }

            float floor = Math.Clamp(gainFloor, 0f, 1f);
            float target = 1f;
            if (echoEnergy > outEnergy * _ratio)
            {
                double echoDominance = echoEnergy / (outEnergy * _ratio);
                double suppression = 1.0 / Math.Sqrt(echoDominance);
                target = MathF.Max(MathF.Max(_minGain, floor), (float)suppression);
            }

            float alpha = target < _gain ? _attack : _release;
            _gain = alpha * _gain + (1f - alpha) * target;
            // 近端确认帧不等慢释放: 平滑增益立即抬到下限, 用户语音当帧就不再被削。
            if (_gain < floor)
                _gain = floor;

            if (_gain >= 0.9995f) return;
            for (int i = 0; i < output.Length; i++)
                output[i] *= _gain;
        }

        public void Reset() => _gain = 1f;
    }
}
