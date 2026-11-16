using System.Collections.Generic;

namespace VoiceBox.Core.Onnx
{
    /// <summary>
    /// CUDA EP 高级调优参数。仅在 <see cref="OnnxExecutionProvider.Cuda"/> 生效；其它 EP 路径完全忽略此对象。
    /// 所有属性的默认值是 ORT 原生 default，但 <see cref="CudnnConvAlgoSearch"/> 设为
    /// <c>HEURISTIC</c>(而非 ORT 的 EXHAUSTIVE)：EXHAUSTIVE 会在首次遇到新形状时
    /// 试运行所有 cuDNN 卷积算法 (秒级耗时 + 显存抖动)；HEURISTIC 启动快、与最优算法几乎无差。
    /// 需要纯 ORT 默认行为时显式设为 <c>"EXHAUSTIVE"</c> / <c>"DEFAULT"</c>。
    /// </summary>
    public sealed class CudaExecutionTuning
    {
        /// <summary>
        /// <c>cudnn_conv_algo_search</c>。可选 <c>EXHAUSTIVE</c> / <c>HEURISTIC</c> / <c>DEFAULT</c>。
        /// 默认 <c>HEURISTIC</c>。
        /// </summary>
        public string CudnnConvAlgoSearch { get; set; } = "HEURISTIC";

        /// <summary>
        /// <c>arena_extend_strategy</c>。可选 <c>kNextPowerOfTwo</c> / <c>kSameAsRequested</c>。
        /// 静态 shape 工作负载 (Qwen3-ASR decode KV cache 等) 用 <c>kSameAsRequested</c> 减少
        /// 显存浪费；动态 shape 保持 <c>kNextPowerOfTwo</c>。默认 ORT 原生 <c>kNextPowerOfTwo</c>。
        /// </summary>
        public string ArenaExtendStrategy { get; set; } = "kNextPowerOfTwo";

        /// <summary>
        /// <c>cudnn_conv_use_max_workspace</c>。默认 true (ORT 1.14+ 默认)。
        /// </summary>
        public bool CudnnConvUseMaxWorkspace { get; set; } = true;

        /// <summary>
        /// <c>do_copy_in_default_stream</c>。设为 false 用独立 copy stream 可获更好 H2D/计算重叠，
        /// 但部分驱动版本会引入数据可见性问题；默认 true 保守。
        /// </summary>
        public bool DoCopyInDefaultStream { get; set; } = true;

        /// <summary>
        /// 单 session 显存上限 (字节)。null 表示不限制 (ORT 默认)。
        /// </summary>
        public long? GpuMemLimitBytes { get; set; }

        /// <summary>
        /// 上述显式选项之外，需要透传给 CUDA EP 的额外 key/value。会覆盖同名显式选项。
        /// </summary>
        public Dictionary<string, string>? AdditionalOptions { get; set; }

        internal Dictionary<string, string> BuildProviderOptionsDictionary(int deviceId)
        {
            var dict = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["device_id"] = deviceId.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["cudnn_conv_algo_search"] = CudnnConvAlgoSearch,
                ["arena_extend_strategy"] = ArenaExtendStrategy,
                ["cudnn_conv_use_max_workspace"] = CudnnConvUseMaxWorkspace ? "1" : "0",
                ["do_copy_in_default_stream"] = DoCopyInDefaultStream ? "1" : "0",
            };
            if (GpuMemLimitBytes is { } limit && limit > 0)
                dict["gpu_mem_limit"] = limit.ToString(System.Globalization.CultureInfo.InvariantCulture);
            if (AdditionalOptions is not null)
                foreach (var kv in AdditionalOptions)
                    dict[kv.Key] = kv.Value;
            return dict;
        }
    }
}
