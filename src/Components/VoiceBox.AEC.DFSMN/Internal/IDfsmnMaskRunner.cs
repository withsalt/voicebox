namespace VoiceBox.AEC.DFSMN.Internal
{
    internal interface IDfsmnMaskRunner : IDisposable
    {
        int FeatureDim { get; }
        int MaskBins { get; }
        bool HasVadProbabilities { get; }
        void Run(ReadOnlySpan<float> features, int frames, Span<float> masks);
        void Run(ReadOnlySpan<float> features, int frames, Span<float> masks, Span<float> vadProbabilities);
        void Reset();
    }
}
