namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// 设备枚举结果中的底层音频主机 API 类型。
    /// 具体支持的 API 取决于当前音频后端与平台。
    /// </summary>
    public enum HostApiType
    {
        /// <summary>
        /// 内部使用，代表一个仍在开发中、尚未完成的后端。通常不会在正式版本中遇到。
        /// </summary>
        InDevelopment = 0,

        /// <summary>
        /// Microsoft Windows 系统专用：DirectSound API。
        /// </summary>
        DirectSound = 1,

        /// <summary>
        /// Microsoft Windows 系统专用：MME (MultiMedia Extensions) API。
        /// 这是 Windows 最早、兼容性最广的音频 API，但延迟较高。
        /// </summary>
        MME = 2,

        /// <summary>
        /// Steinberg 公司的 ASIO (Audio Stream Input/Output) API。
        /// 在 Windows 上提供专业级的低延迟音频录制和播放，需要安装专门的 ASIO 驱动程序。
        /// </summary>
        ASIO = 3,

        /// <summary>
        /// 旧版 Apple Mac OS (Classic Mac OS 9 及更早版本) 的 Sound Manager API。
        /// 现在已被 CoreAudio 取代。
        /// </summary>
        SoundManager = 4,

        /// <summary>
        /// Apple macOS 和 iOS 系统专用：Core Audio API。
        /// 这是 Apple 平台上现今标准的低延迟音频架构。
        /// </summary>
        CoreAudio = 5,

        /// <summary>
        /// Linux/Unix 系统专用：OSS (Open Sound System) API。
        /// 这是 Linux 早期的标准音频接口，现今大多被 ALSA 取代。
        /// </summary>
        OSS = 7,

        /// <summary>
        /// Linux 系统专用：ALSA (Advanced Linux Sound Architecture) API。
        /// 这是目前 Linux 内核中标准的音频架构。
        /// </summary>
        ALSA = 8,

        /// <summary>
        /// SGI (Silicon Graphics) IRIX 系统上的 AL (Audio Library) API。
        /// 相当罕见。
        /// </summary>
        AL = 9,

        /// <summary>
        /// BeOS 操作系统的音频 API。
        /// BeOS 是一款历史悠久的多媒体操作系统。
        /// </summary>
        BeOS = 10,

        /// <summary>
        /// Microsoft Windows 系统专用：WDM-KS (Windows Driver Model - Kernel Streaming) API。
        /// 允许直接与内核层的音频驱动程序通信，可以实现非常低的延迟。
        /// </summary>
        WDMKS = 11,

        /// <summary>
        /// JACK (JACK Audio Connection Kit) 服务器。
        /// 一个在 Linux、macOS 和 Windows 上都可用的专业音频服务器，专为低延迟和灵活的应用程序间音频路由而设计。
        /// </summary>
        JACK = 12,

        /// <summary>
        /// Microsoft Windows 系统专用：WASAPI (Windows Audio Session API)。
        /// 从 Windows Vista 开始引入，是目前 Windows 系统上建议使用的标准音频 API，提供低延迟的“独占模式”(Exclusive Mode)。
        /// </summary>
        WASAPI = 13,

        /// <summary>
        /// AudioScience Inc. 公司硬件的 HPI (Hardware Programming Interface)。
        /// 用于控制其专业音频卡。
        /// </summary>
        AudioScienceHPI = 14,

        /// <summary>
        /// 苹果公司为 iOS 设计的 AudioIO API，基于 CoreAudio。
        /// </summary>
        AudioIO = 15,

        /// <summary>
        /// Linux 桌面环境中常见的 PulseAudio 音频服务器。
        /// 提供进阶的音量控制、网络音频和多应用程序混音功能。
        /// </summary>
        PulseAudio = 16,

        /// <summary>
        /// OpenBSD 操作系统的 Sndio 音频服务器，也适用于 Linux 和其他 BSD 系统。
        /// </summary>
        Sndio = 17,

        /// <summary>
        /// NetBSD/OpenBSD audio(4) backend.
        /// </summary>
        Audio4 = 18,

        /// <summary>
        /// Android AAudio backend.
        /// </summary>
        AAudio = 19,

        /// <summary>
        /// Android OpenSL ES backend.
        /// </summary>
        OpenSL = 20,

        /// <summary>
        /// Web Audio backend.
        /// </summary>
        WebAudio = 21,

        /// <summary>
        /// A custom or null audio backend.
        /// </summary>
        Other = 22,
    }
}
