namespace VoiceBox.Core.Services.AudioService.Models
{
    [Flags]
    public enum AudioDeviceType
    {
        /// <summary>
        /// 未知或未指定类型。
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// 输入设备（如麦克风）。
        /// </summary>
        Input = 1 << 0, // 2^0

        /// <summary>
        /// 输出设备（如扬声器、耳机）。
        /// </summary>
        Output = 1 << 1, // 2^1

        /// <summary>
        /// 表示所有类型的设备。
        /// </summary>
        InputOutput = 1 << 2
    }
}
