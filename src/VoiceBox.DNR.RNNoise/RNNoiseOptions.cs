using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.DNR.RNNoise
{
    /// <summary>
    /// RNNoise 实时降噪服务配置。
    /// RNNoise 原生工作于 48kHz 单声道，输入采样率不为 48000 时会内部自动重采样。
    /// </summary>
    public class RNNoiseOptions
    {
        /// <summary>
        /// 输入音频采样率 (Hz)。默认 16000。
        /// 内部会与 RNNoise 原生工作率 48000 进行重采样转换。
        /// </summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>
        /// 声道数。RNNoise 仅支持单声道处理，当前实现要求该值为 1。
        /// </summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// 采样格式。API 层固定使用 Float32（与 IDigitalNoiseReductionService 接口一致），
        /// 此处保留仅为与其它服务配置形式一致。
        /// </summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// 自定义 RNNoise 模型文件路径 (.rnnn)。为空时使用 RNNoise 内置默认模型。
        /// </summary>
        public string? ModelPath { get; set; }

        /// <summary>
        /// 是否旁路 (不做降噪直接透传)。默认 false，主要用于 A/B 对比测试。
        /// </summary>
        public bool Bypass { get; set; } = false;
    }
}
