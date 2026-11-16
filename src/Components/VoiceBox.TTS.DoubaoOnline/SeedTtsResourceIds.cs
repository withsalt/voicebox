namespace VoiceBox.TTS.DoubaoOnline
{
    /// <summary>
    /// 火山引擎语音合成的资源信息 ID (<c>X-Api-Resource-Id</c>)，用于声明调用的模型版本 / 计费类型。
    /// 详见文档「双向流式语音合成 WebSocket」的「请求头」章节。
    /// </summary>
    public static class SeedTtsResourceIds
    {
        /// <summary>
        /// 豆包语音合成大模型 2.0。支持使用「豆包语音合成模型 2.0 音色」，
        /// 可直接使用语音指令 (context_texts)。
        /// </summary>
        public const string SeedTts20 = "seed-tts-2.0";

        /// <summary>
        /// 豆包声音复刻大模型 2.0。支持使用声音复刻接口克隆的音色。
        /// 复刻音色暂不支持语音指令 (context_texts)。
        /// </summary>
        public const string SeedIcl20 = "seed-icl-2.0";
    }
}
