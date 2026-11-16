using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.Core.Onnx
{
    /// <summary>
    /// 统一的 ONNX <see cref="SessionOptions"/> 构造工厂。
    /// 各 ONNX 推理项目应通过 <see cref="Create(OnnxExecutionOptions)"/> 获取 SessionOptions，
    /// 而不是各自重复实现 EP / 线程等设置。
    /// </summary>
    public static class OnnxSessionFactory
    {
        /// <summary>
        /// 按 <paramref name="options"/> 配置创建 <see cref="SessionOptions"/>。
        /// 若 <see cref="OnnxExecutionOptions.SessionOptionsFactory"/> 已设置则优先调用该工厂。
        /// </summary>
        public static SessionOptions Create(OnnxExecutionOptions options)
            => Create(options, SessionPurpose.Dynamic);

        /// <summary>
        /// 同 <see cref="Create(OnnxExecutionOptions)"/>，但允许调用方按 session 用途
        /// 选择不同的 MemoryPattern / 形状策略。
        /// </summary>
        public static SessionOptions Create(OnnxExecutionOptions options, SessionPurpose purpose)
        {
            ArgumentNullException.ThrowIfNull(options);

            if (options.SessionOptionsFactory is not null)
                return options.SessionOptionsFactory();

            return options.ExecutionProvider switch
            {
                OnnxExecutionProvider.Cuda => CreateCuda(options.GpuDeviceId, options.ThreadsNum, options.CudaTuning, purpose),
                OnnxExecutionProvider.DirectML => CreateDirectMl(options.GpuDeviceId, options.ThreadsNum, purpose),
                _ => CreateCpu(options.ThreadsNum),
            };
        }

        /// <summary>
        /// 创建 CPU 执行的 SessionOptions。<paramref name="threadsNum"/> ≤ 0 时取 CPU 核数的一半。
        /// </summary>
        public static SessionOptions CreateCpu(int threadsNum)
        {
            int t = threadsNum > 0 ? threadsNum : Math.Max(1, Environment.ProcessorCount / 2);
            return new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = t,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableMemoryPattern = true,
                EnableCpuMemArena = true,
            };
        }

        /// <summary>
        /// 创建 CUDA 执行的 SessionOptions。需引用 Microsoft.ML.OnnxRuntime.Gpu。
        /// 兼容旧调用 (无 tuning / purpose) — 等价于动态 shape + ORT 默认 provider 选项。
        /// </summary>
        public static SessionOptions CreateCuda(int deviceId, int threadsNum)
            => CreateCuda(deviceId, threadsNum, tuning: null, purpose: SessionPurpose.Dynamic);

        /// <summary>
        /// 创建 CUDA 执行的 SessionOptions，可传入 <see cref="CudaExecutionTuning"/> 高级 provider
        /// 选项 (cudnn_conv_algo_search 等) 与 <paramref name="purpose"/> 指示形状是否静态。
        /// 关闭 memory pattern 是 dynamic-shape 的默认选择 (避免 CUDA 上预分配过大);
        /// static-shape 用途则打开，让 ORT 复用中间张量。
        /// </summary>
        public static SessionOptions CreateCuda(int deviceId, int threadsNum, CudaExecutionTuning? tuning, SessionPurpose purpose)
        {
            int t = threadsNum > 0 ? threadsNum : Math.Max(1, Environment.ProcessorCount / 2);
            var so = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                IntraOpNumThreads = t,
                InterOpNumThreads = 1,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                EnableMemoryPattern = purpose == SessionPurpose.StaticShapes,
                EnableCpuMemArena = true,
            };

            // Diagnostic: ORT_LOG_VERBOSE=1 surfaces the per-node EP assignment + Memcpy insertion
            // details (which nodes force a CPU↔GPU boundary). Off by default.
            if (Environment.GetEnvironmentVariable("ORT_LOG_VERBOSE") == "1")
            {
                so.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
                so.LogVerbosityLevel = 1;
            }

            if (tuning is null)
            {
                // 旧路径 — 不设置任何 CUDA provider option，沿用 ORT 默认行为。
                so.AppendExecutionProvider_CUDA(deviceId);
            }
            else
            {
                using var providerOptions = new OrtCUDAProviderOptions();
                providerOptions.UpdateOptions(tuning.BuildProviderOptionsDictionary(deviceId));
                so.AppendExecutionProvider_CUDA(providerOptions);
            }
            return so;
        }

        /// <summary>
        /// 创建 DirectML 执行的 SessionOptions (Windows GPU)。需引用 Microsoft.ML.OnnxRuntime.DirectML。
        /// </summary>
        public static SessionOptions CreateDirectMl(int deviceId, int threadsNum)
            => CreateDirectMl(deviceId, threadsNum, SessionPurpose.Dynamic);

        /// <summary>同 <see cref="CreateDirectMl(int, int)"/>，但允许指定 <paramref name="purpose"/>。</summary>
        public static SessionOptions CreateDirectMl(int deviceId, int threadsNum, SessionPurpose purpose)
        {
            int t = threadsNum > 0 ? threadsNum : Math.Max(1, Environment.ProcessorCount / 2);
            var so = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                EnableMemoryPattern = purpose == SessionPurpose.StaticShapes,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                IntraOpNumThreads = t,
                InterOpNumThreads = 1,
            };
            so.AppendExecutionProvider_DML(deviceId);
            so.AppendExecutionProvider_CPU();
            return so;
        }
    }
}
