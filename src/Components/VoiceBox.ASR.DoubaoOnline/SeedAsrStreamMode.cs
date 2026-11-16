namespace VoiceBox.ASR.DoubaoOnline
{
    /// <summary>
    /// 火山引擎流式语音识别的接入模式，对应不同的 WebSocket 接口地址。
    /// </summary>
    public enum SeedAsrStreamMode
    {
        /// <summary>
        /// 双向流式模式 (<c>/api/v3/sauc/bigmodel</c>)。每输入一包返回一包，
        /// 尽快返回识别到的字符，首字时延低。
        /// </summary>
        Bidirectional = 0,

        /// <summary>
        /// 双向流式模式 (<c>/api/v3/sauc/bigmodel_async</c>)。仅当结果发生变化时才下发新数据包；
        /// 开启二遍识别 (enable_nonstream) 也走此接口。
        /// </summary>
        BidirectionalV2 = 1,

        /// <summary>
        /// 流式输入模式 (<c>/api/v3/sauc/bigmodel_nostream</c>)。音频超过 15s 或收到最后一包后
        /// 才返回结果，准确率更高但时延更大。
        /// </summary>
        StreamingInput = 2,
    }

    internal static class SeedAsrStreamModeExtensions
    {
        /// <summary>返回该模式对应的 WebSocket 请求路径。</summary>
        public static string ToRequestPath(this SeedAsrStreamMode mode) => mode switch
        {
            SeedAsrStreamMode.Bidirectional => "/api/v3/sauc/bigmodel",
            SeedAsrStreamMode.BidirectionalV2 => "/api/v3/sauc/bigmodel_async",
            SeedAsrStreamMode.StreamingInput => "/api/v3/sauc/bigmodel_nostream",
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown SeedAsrStreamMode."),
        };
    }
}
