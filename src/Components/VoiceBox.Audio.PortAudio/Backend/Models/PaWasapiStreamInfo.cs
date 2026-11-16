using System.Runtime.InteropServices;

namespace VoiceBox.Audio.PortAudio.Backend.Models
{
    /// <summary>
    /// WASAPI 专用的打开流参数 (对应 pa_win_wasapi.h 的 PaWasapiStreamInfo, v19.7.0 布局)。
    /// <para>
    /// 原生侧按 size == sizeof(PaWasapiStreamInfo) &amp;&amp; version == 1 &amp;&amp; hostApiType == paWASAPI
    /// 严格校验, 字段顺序与类型不可更改; 该版本不含 19.7 之后新增的 passthroughDetails 字段。
    /// </para>
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PaWasapiStreamInfo
    {
        /// <summary>sizeof(PaWasapiStreamInfo), 原生侧用于版本兼容校验。</summary>
        public uint size;

        /// <summary>固定为 paWASAPI (13)。</summary>
        public int hostApiType;

        /// <summary>固定为 1。</summary>
        public uint version;

        /// <summary><see cref="PaWasapiFlags"/> 的组合。</summary>
        public uint flags;

        /// <summary>WAVEFORMATEXTENSIBLE 声道掩码, 仅 UseChannelMask 标志时生效。</summary>
        public uint channelMask;

        /// <summary>宿主处理回调 (RedirectHostProcessor 标志时生效), 不使用时为 0。</summary>
        public nint hostProcessorOutput;
        public nint hostProcessorInput;

        /// <summary>PaWasapiThreadPriority, 仅 ThreadPriority 标志时生效。</summary>
        public int threadPriority;

        /// <summary>PaWasapiStreamCategory。</summary>
        public int streamCategory;

        /// <summary>PaWasapiStreamOption。</summary>
        public int streamOption;
    }

    /// <summary>
    /// pa_win_wasapi.h 的 PaWasapiThreadPriority: PortAudio WASAPI 处理线程注册到 MMCSS 的任务类别。
    /// 仅在 <see cref="PaWasapiFlags.ThreadPriority"/> 标志置位时生效。
    /// </summary>
    internal enum PaWasapiThreadPriority
    {
        None = 0,
        Audio = 1,
        Capture = 2,
        Distribution = 3,
        Games = 4,
        Playback = 5,
        ProAudio = 6,
        WindowManager = 7,
    }

    /// <summary>pa_win_wasapi.h 的 PaWasapiFlags。</summary>
    [Flags]
    internal enum PaWasapiFlags : uint
    {
        None = 0,

        /// <summary>独占模式 (最低延迟, 但独占设备且要求设备原生支持请求的格式)。</summary>
        Exclusive = 1 << 0,

        /// <summary>跳过 PortAudio 内部处理, 由 hostProcessor 回调直接收发数据。</summary>
        RedirectHostProcessor = 1 << 1,

        /// <summary>使用 channelMask 指定扬声器布局。</summary>
        UseChannelMask = 1 << 2,

        /// <summary>使用轮询而非事件驱动 (延迟更高, 仅特殊场景)。</summary>
        Polling = 1 << 3,

        /// <summary>使用 threadPriority 字段显式指定线程优先级。</summary>
        ThreadPriority = 1 << 4,

        /// <summary>禁止 PortAudio 自行挑选工作采样格式, 格式不支持时直接报错。</summary>
        ExplicitSampleFormat = 1 << 5,

        /// <summary>
        /// 允许系统插入声道矩阵混音与重采样器, 使流格式 (例如 16kHz) 可以与系统混音格式
        /// (通常 44.1/48kHz) 不一致。仅共享模式生效; 不设置时格式不匹配会直接打不开流。
        /// </summary>
        AutoConvert = 1 << 6,
    }
}
