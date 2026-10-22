namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class MaskedStftProcessor
    {
        private readonly int _frameLength;
        private readonly int _hopLength;
        private readonly int _bins;
        private readonly float _maskFloor;
        private readonly float[] _window;
        private readonly RealDft _dft;
        private readonly float[] _analysisFrame;
        private readonly float[] _re;
        private readonly float[] _im;
        private readonly float[] _timeFrame;
        private readonly float[] _ola;
        private readonly float[] _norm;

        public MaskedStftProcessor(int frameLength, int hopLength, float? maskFloorDb)
        {
            if (frameLength <= 0 || (frameLength & 1) != 0) throw new ArgumentOutOfRangeException(nameof(frameLength));
            if (hopLength <= 0 || hopLength > frameLength) throw new ArgumentOutOfRangeException(nameof(hopLength));

            _frameLength = frameLength;
            _hopLength = hopLength;
            _bins = frameLength / 2 + 1;
            _maskFloor = maskFloorDb is { } db ? DspMath.AttenuationDbToGain(db) : 0f;
            _window = DspMath.HammingWindow(frameLength);
            _dft = new RealDft(frameLength);
            _analysisFrame = new float[frameLength];
            _re = new float[_bins];
            _im = new float[_bins];
            _timeFrame = new float[frameLength];
            _ola = new float[frameLength];
            _norm = new float[frameLength];
        }

        public void ProcessFrame(ReadOnlySpan<float> linearFrame, ReadOnlySpan<float> mask, Span<float> outputHop)
            => ProcessFrame(linearFrame, mask, outputHop, _maskFloor);

        /// <summary>
        /// <paramref name="maskFloor"/> 覆盖本帧的掩码下限 (线性域)。近端保护在双讲帧抬高该
        /// 下限, 限制掩码可施加的最大抑制; 传构造时的默认值则与旧行为逐位一致。
        /// </summary>
        public void ProcessFrame(ReadOnlySpan<float> linearFrame, ReadOnlySpan<float> mask, Span<float> outputHop, float maskFloor)
        {
            if (linearFrame.Length != _frameLength)
                throw new ArgumentException($"linearFrame length must be {_frameLength}.", nameof(linearFrame));
            if (mask.Length != _bins)
                throw new ArgumentException($"mask length must be {_bins}.", nameof(mask));
            if (outputHop.Length != _hopLength)
                throw new ArgumentException($"outputHop length must be {_hopLength}.", nameof(outputHop));

            for (int i = 0; i < _frameLength; i++)
                _analysisFrame[i] = linearFrame[i] * _window[i];

            _dft.Forward(_analysisFrame, _re, _im);
            for (int k = 0; k < _bins; k++)
            {
                float m = DspMath.Clamp(mask[k], maskFloor, 1.5f);
                _re[k] *= m;
                _im[k] *= m;
            }

            _dft.Inverse(_re, _im, _timeFrame);
            for (int i = 0; i < _frameLength; i++)
            {
                float w = _window[i];
                _ola[i] += _timeFrame[i] * w;
                _norm[i] += w * w;
            }

            for (int i = 0; i < _hopLength; i++)
            {
                float denom = _norm[i];
                outputHop[i] = denom > 1e-8f ? _ola[i] / denom : 0f;
            }

            int keep = _frameLength - _hopLength;
            Array.Copy(_ola, _hopLength, _ola, 0, keep);
            Array.Clear(_ola, keep, _hopLength);
            Array.Copy(_norm, _hopLength, _norm, 0, keep);
            Array.Clear(_norm, keep, _hopLength);
        }

        public void Reset()
        {
            Array.Clear(_ola);
            Array.Clear(_norm);
        }
    }
}
