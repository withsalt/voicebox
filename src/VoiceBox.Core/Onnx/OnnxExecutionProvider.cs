namespace VoiceBox.Core.Onnx
{
    /// <summary>
    /// ONNX Runtime 执行提供者 (CPU / CUDA / DirectML)。
    /// </summary>
    public enum OnnxExecutionProvider
    {
        /// <summary>CPU 执行 (默认)。无需额外 NuGet 包。</summary>
        Cpu = 0,

        /// <summary>NVIDIA CUDA GPU。需引用 Microsoft.ML.OnnxRuntime.Gpu 包。</summary>
        Cuda = 1,

        /// <summary>DirectML GPU (Windows)。需引用 Microsoft.ML.OnnxRuntime.DirectML 包。</summary>
        DirectML = 2,
    }
}
