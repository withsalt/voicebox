namespace VoiceBox.ASR.DoubaoOnline
{
    /// <summary>
    /// 火山引擎语音识别的资源信息 ID (<c>X-Api-Resource-Id</c>)，用于声明调用的计费/服务类型。
    /// 详见文档「鉴权」章节。
    /// </summary>
    public static class SeedAsrResourceIds
    {
        /// <summary>豆包流式语音识别模型 1.0 - 小时版。</summary>
        public const string BigAsrDuration = "volc.bigasr.sauc.duration";

        /// <summary>豆包流式语音识别模型 1.0 - 并发版。</summary>
        public const string BigAsrConcurrent = "volc.bigasr.sauc.concurrent";

        /// <summary>豆包流式语音识别模型 2.0 - 小时版。</summary>
        public const string SeedAsrDuration = "volc.seedasr.sauc.duration";

        /// <summary>豆包流式语音识别模型 2.0 - 并发版。</summary>
        public const string SeedAsrConcurrent = "volc.seedasr.sauc.concurrent";
    }
}
