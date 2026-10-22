namespace VoiceBox.Core.Onnx
{
    public static class OnnxExecutionProviderExtensions
    {
        /// <summary>
        /// Maps a standard ONNX Runtime execution provider to the corresponding GenAI provider.
        /// </summary>
        public static GenAIExecutionProvider ToGenAIExecutionProvider(this OnnxExecutionProvider provider)
        {
            return provider switch
            {
                OnnxExecutionProvider.Cpu => GenAIExecutionProvider.Cpu,
                OnnxExecutionProvider.Cuda => GenAIExecutionProvider.Cuda,
                OnnxExecutionProvider.DirectML => GenAIExecutionProvider.DirectML,
                _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, null),
            };
        }
    }
}
