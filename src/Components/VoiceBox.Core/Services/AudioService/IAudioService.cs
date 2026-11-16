using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Core.Services.AudioService
{
    public interface IAudioService : IDisposable
    {
        /// <summary>
        /// 当前音频服务的配置。
        /// </summary>
        AudioOptionsBase Options { get; }

        /// <summary>
        /// 列出所有可用的输入音频设备。
        /// </summary>
        /// <param name="deviceType">设备类型</param>
        /// <param name="enumAllDevices">是否枚举所有后端API设备，默认只列出当前使用默认设备所在驱动设备</param>
        /// <returns>音频设备信息列表</returns>
        IReadOnlyList<AudioDevice> ListAudioDevices(AudioDeviceType? deviceType = null, bool enumAllDevices = false);
    }
}
