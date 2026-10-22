namespace VoiceBox.Core.Onnx
{
    /// <summary>
    /// ONNX Runtime GenAI 的执行提供者。
    /// 与 <see cref="OnnxExecutionProvider"/> 是平行枚举:
    /// 后者用于 Microsoft.ML.OnnxRuntime (普通推理); 这里用于 Microsoft.ML.OnnxRuntimeGenAI (生成式推理),
    /// 两者的可选项不完全对等 (例如 GenAI 还可走 QNN/WebGPU)。
    /// </summary>
    public enum GenAIExecutionProvider
    {
        /// <summary>遵循 <c>genai_config.json</c> 中已配置的 provider, 不做覆盖。</summary>
        FollowConfig = 0,

        /// <summary>CPU。无附加 NuGet 依赖。</summary>
        Cpu = 1,

        /// <summary>NVIDIA CUDA GPU。需要 <c>Microsoft.ML.OnnxRuntimeGenAI.Cuda</c> 原生包。</summary>
        Cuda = 2,

        /// <summary>DirectML GPU (Windows)。需要 <c>Microsoft.ML.OnnxRuntimeGenAI.DirectML</c> 原生包。</summary>
        DirectML = 3,
    }
}
