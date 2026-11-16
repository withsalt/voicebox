using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.DNR.Hush.Models
{
    /// <summary>
    /// Hush 实时降噪服务配置。
    ///
    /// Hush（DeepFilterNet3-SE）仅支持 16 kHz 单声道，按 10 ms (hop=160 sample) 的帧逐帧
    /// 流式处理，通过外置 GRU 状态的有状态 ONNX 实现真·逐帧低延迟。输入采样率不为 16000 时
    /// 服务会直接拒绝。
    /// </summary>
    public class HushOptions : OnnxExecutionOptions
    {
        /// <summary>
        /// Hush 每次只跑一个 10 ms 流式帧。小线程池可避免在与 AEC/VAD 同时运行的
        /// 实时管线中过度占用 CPU 核。
        /// </summary>
        public override int ThreadsNum { get; set; } = 2;

        /// <summary>
        /// 输入音频采样率 (Hz)。必须为 16000（Hush 原生工作率）。
        /// 非 16000 时服务构造会抛出 <see cref="NotSupportedException"/>。
        /// </summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>声道数。Hush 仅支持单声道处理，要求该值为 1。</summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// 采样格式。API 层固定使用 Float32 (与 IDigitalNoiseReductionService 接口一致)，
        /// 此处保留仅为与其它服务配置形式一致。
        /// </summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>选用的模型档位。默认 <see cref="HushModel.Hush16k"/>。</summary>
        public HushModel Model { get; set; } = HushModel.Hush16k;

        /// <summary>
        /// 自定义模型目录。为空时按 <see cref="Model"/> 从内置 runtimes 目录加载。
        /// 该目录须包含 <c>combined_streaming.onnx</c> 与 <c>config.ini</c>。
        /// </summary>
        public string? ModelDir { get; set; }

        /// <summary>
        /// 编码器卷积上下文帧数 T（每帧喂给 ONNX 的时间帧数）。为 null 时取模型 config 的
        /// conv_kernel_inp 首值（卷积时间核大小，Hush 为 3）。增大可为卷积提供更多上下文
        /// （质量略升、算力略增），但 GRU 始终只处理末帧，故不影响因果性与状态连续性。
        /// </summary>
        public int? EncoderContextFrames { get; set; }

    }
}
