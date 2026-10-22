namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// 音频设备信息的简单数据传输对象。
    /// </summary>
    /// <param name="Id">设备索引ID</param>
    /// <param name="Name">设备名称</param>
    /// <param name="MaxInputChannels">最大输入声道数</param>
    /// <param name="DefaultSampleRate">默认采样率</param>
    public record AudioDevice(int Id, string Name, bool IsDefault, int MaxInputChannels, int MaxOutputChannels, double DefaultSampleRate, AudioDeviceType DeviceType, HostApiType HostApi);
}
