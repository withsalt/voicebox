using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.DNR.DPDFNet.Models
{
    /// <summary>
    /// DPDFNet 实时降噪服务配置。
    ///
    /// DPDFNet 原生工作于 16 kHz 单声道，按 10 ms (hop=160 sample) 的帧流式处理。
    /// 当前服务要求输入采样率必须为 16000。
    /// </summary>
    public class DPDFNetOptions : OnnxExecutionOptions
    {
        /// <summary>
        /// DPDFNet runs one 10 ms streaming frame at a time. A small ORT thread pool avoids
        /// oversubscribing CPU cores when it is combined with AEC/VAD in live pipelines.
        /// </summary>
        public override int ThreadsNum { get; set; } = 2;

        /// <summary>
        /// 输入音频采样率 (Hz)。必须为 16000（DPDFNet 原生工作率）。
        /// </summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>
        /// 声道数。DPDFNet 仅支持单声道处理，当前实现要求该值为 1。
        /// </summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// 采样格式。API 层固定使用 Float32 (与 IDigitalNoiseReductionService 接口一致)，
        /// 此处保留仅为与其它服务配置形式一致。
        /// </summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// 选用的模型档位。默认 <see cref="DPDFNetModel.DpDfNet2"/>。
        /// 当显式设置 <see cref="ModelPath"/> 时，此项被忽略。
        /// </summary>
        public DPDFNetModel Model { get; set; } = DPDFNetModel.DpDfNet2;

        /// <summary>
        /// 自定义模型文件路径 (.onnx)。为空时按 <see cref="Model"/> 从内置 runtimes 目录加载。
        /// </summary>
        public string? ModelPath { get; set; }

        /// <summary>
        /// 衰减上限 (dB)，限制对每个频点的最大抑制量，用于在强降噪与音质间折中。
        /// <list type="bullet">
        ///   <item><c>null</c> (默认)：不做限制，完全采用模型输出。</item>
        ///   <item>≥0：将增强谱与带噪谱按上限混合，0 dB 等价完全旁路，值越大允许的抑制越强。</item>
        /// </list>
        /// </summary>
        public float? AttenuationLimitDb { get; set; }

        /// <summary>
        /// <see cref="AttenuationLimitDb"/> 的施加方式, 默认 <see cref="DPDFNetAttenuationMode.MagnitudeFloor"/>可避免复数混合的相位伪影，对下游 ASR 更友好
        /// </summary>
        public DPDFNetAttenuationMode AttenuationMode { get; set; } = DPDFNetAttenuationMode.MagnitudeFloor;
    }
}
