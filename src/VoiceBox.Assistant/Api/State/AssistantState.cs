namespace VoiceBox.Assistant.Api.State
{
    /// <summary>
    /// 语音助手的对外状态。
    /// </summary>
    public enum AssistantState
    {
        /// <summary>就绪, 等待用户开口。</summary>
        Idle = 0,

        /// <summary>VAD 检测到说话开始, 正在录音并喂给 ASR。</summary>
        Listening = 1,

        /// <summary>VAD 检测到说话结束, 正在等待 ASR 完成识别。</summary>
        Recognizing = 2,

        /// <summary>已拿到识别文本, 正在等待 LLM 首句。</summary>
        Thinking = 3,

        /// <summary>正在合成并播放 TTS 音频; 启用可打断时此状态仍会监听用户输入。</summary>
        Speaking = 4,

        /// <summary>已停止, 不再响应任何事件。</summary>
        Stopped = 5,
    }
}
