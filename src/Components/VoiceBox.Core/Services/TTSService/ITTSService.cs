using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.Core.Services.TTSService
{
    public interface ITTSService : IDisposable
    {
        /// <summary>
        /// 模型实际输出的采样率 (Hz)。配置 <see cref="AudioService.IAudioPlayService"/> 时应当用这个值。
        /// </summary>
        int OutputSampleRate { get; }

        /// <summary>
        /// 模型实际输出的声道数。
        /// </summary>
        int OutputChannels { get; }

        IReadOnlyList<TtsVoice> ListVoices();

        /// <summary>
        /// 获取当前服务配置解析后的默认音色。没有可用音色时返回 null。
        /// </summary>
        TtsVoice? GetDefaultVoice();

        Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default);
    }
}
