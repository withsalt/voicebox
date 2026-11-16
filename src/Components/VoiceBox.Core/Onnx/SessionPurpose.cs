namespace VoiceBox.Core.Onnx
{
    /// <summary>
    /// 一个 ONNX session 的"用途"提示，决定 SessionOptions 的细节调优 (主要是 MemoryPattern)。
    /// 编码器 / prefill 这类 shape 随调用变化的图保留 <see cref="Dynamic"/>；
    /// 单 token decode 这类 shape 全静态的图改用 <see cref="StaticShapes"/>。CPU EP 下两者行为相同。
    /// </summary>
    public enum SessionPurpose
    {
        /// <summary>形状会随调用变化 (默认)。CUDA/DML 下保守地关闭 MemoryPattern。</summary>
        Dynamic = 0,

        /// <summary>形状完全静态 (autoregressive decode 之类)。允许 MemoryPattern 复用中间张量。</summary>
        StaticShapes = 1,
    }
}
