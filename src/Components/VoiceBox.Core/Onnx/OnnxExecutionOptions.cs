using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.Core.Onnx
{
    /// <summary>
    /// 公共 ONNX 推理执行配置基类。各服务的 Options 通过继承此类获得统一的 EP/线程/工厂配置。
    /// </summary>
    public abstract class OnnxExecutionOptions
    {
        /// <summary>执行提供者 (CPU / CUDA / DirectML)。</summary>
        public OnnxExecutionProvider ExecutionProvider { get; set; } = OnnxExecutionProvider.Cpu;

        /// <summary>GPU 设备号 (Cuda / DirectML 用)。</summary>
        public int GpuDeviceId { get; set; } = 0;

        /// <summary>ONNX intra-op 线程数。≤0 表示自动 (CPU 核数的一半)。</summary>
        public virtual int ThreadsNum { get; set; } = 0;

        /// <summary>
        /// CUDA EP 高级选项。仅在 <see cref="ExecutionProvider"/> 为 <see cref="OnnxExecutionProvider.Cuda"/>
        /// 时生效；null 表示沿用 ORT 默认 (旧行为)。各服务可在自身 Options 构造里设置默认值。
        /// </summary>
        public CudaExecutionTuning? CudaTuning { get; set; }

        /// <summary>
        /// 自定义 SessionOptions 工厂。设置后优先级高于 <see cref="ExecutionProvider"/>。
        /// </summary>
        public Func<SessionOptions>? SessionOptionsFactory { get; set; }
    }
}
