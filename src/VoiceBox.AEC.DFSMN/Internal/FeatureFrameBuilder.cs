namespace VoiceBox.AEC.DFSMN.Internal
{
    internal sealed class FeatureFrameBuilder
    {
        private readonly DfsmnFbankExtractor _fbank;
        private readonly MvnTransform _mvn;
        private readonly int _frameLength;
        private readonly int _bins;
        private readonly float[] _scratch;

        public int FeatureDim => _bins * 3;

        public FeatureFrameBuilder(int frameLength, int bins, MvnTransform mvn)
        {
            _frameLength = frameLength;
            _bins = bins;
            _mvn = mvn;
            if (mvn.Dimension != FeatureDim)
                throw new ArgumentException($"MVN dimension must be {FeatureDim}.", nameof(mvn));

            _fbank = new DfsmnFbankExtractor(frameLength, bins);
            _scratch = new float[frameLength];
        }

        public void Build(
            ReadOnlySpan<float> near,
            ReadOnlySpan<float> linear,
            ReadOnlySpan<float> echo,
            Span<float> destination)
        {
            if (near.Length != _frameLength || linear.Length != _frameLength || echo.Length != _frameLength)
                throw new ArgumentException($"all input frames must have length {_frameLength}.");
            if (destination.Length != FeatureDim)
                throw new ArgumentException($"destination length must be {FeatureDim}.", nameof(destination));

            ComputeBranch(near, destination[.._bins]);
            ComputeBranch(linear, destination.Slice(_bins, _bins));
            ComputeBranch(echo, destination.Slice(_bins * 2, _bins));
            _mvn.Apply(destination);
        }

        private void ComputeBranch(ReadOnlySpan<float> sourceUnitScale, Span<float> destination)
        {
            for (int i = 0; i < _frameLength; i++)
                _scratch[i] = DspMath.Clamp(sourceUnitScale[i], -1f, 1f) * 32768f;
            _fbank.ComputeFrame(_scratch, destination);
        }
    }
}
