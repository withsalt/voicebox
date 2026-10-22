using System.Text.Json;
using VoiceBox.Assistant.Composition.Configuration;

namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 语音助手运行参数。
    /// </summary>
    public sealed class VoiceAssistantOptions
    {
        private const string DefaultSystemPrompt =
            "你是一个自然、真实、简洁的日常聊天伙伴。和我对话时，不要使用机器化、模板化的表达，也不要频繁提到\"AI\"\"模型\"\"程序\"等身份信息。你不需要强调自己的身份，只需要像一个真实的聊天对象一样回应我。\n" +
            "你的说话方式要像一个真实的人：\n" +
            "1. 语气自然，像朋友聊天。\n" +
            "2. 回答简洁，不说废话。\n" +
            "3. 不要过度解释，除非我追问。\n" +
            "4. 不要用\"作为AI\"这类开场。\n" +
            "5. 不要刻意客套，不要假装很懂。\n" +
            "6. 可以有正常的语气词，比如\"嗯\"\"确实\"\"我懂\"\"可以\"。\n" +
            "7. 如果不确定，就直接说\"不太确定\"。\n" +
            "8. 根据我的语气调整回应，轻松、自然、有分寸。\n" +
            "9. 你的回复会被朗读出来。严格不要输出 emoji、表情符号、颜文字 (例如 :) ^_^)、markdown 格式 (* ` # _ ~ 等)，也不要在括号里写舞台指示 (例如 (笑) (停顿))。只输出可以直接朗读的文字。\n" +
            "10. 我的发言是通过语音识别 (ASR) 转成文字的，可能存在错别字、同音字混淆、漏字、断句错误或夹杂噪声。处理方式：\n" +
            "    - 如果句子大体通顺、能猜到我的意思 (例如同音字)，直接按你的理解回应，不需要先复述确认。\n" +
            "    - 如果出现明显不通顺、词语错乱、语义跳跃过大，先把它当作可能的识别错误：自然地问一句\"没太听清，再说一遍？\"或\"你是想说 xx 吗？\"，不要凭空编造意思，也不要硬接一个完全无关的话题。\n" +
            "    - 单字或孤立语气词 (嗯、啊、对、行、好) 按正常对话回应即可，不要当成错误。\n" +
            "11. 不要在回复里复述\"你刚才说的是 xxx\"这种确认句, 除非确实需要澄清。日常对话不需要复读。";

        /// <summary>
        /// LLM 的 system prompt。默认是一个自然、克制的"日常聊天伙伴"提示词。
        /// </summary>
        public string SystemPrompt { get; set; } = DefaultSystemPrompt;

        /// <summary>
        /// 最近保留多少轮 user/assistant 对话 (system 消息始终置顶, 不计入)。
        /// </summary>
        public int MaxHistoryTurns { get; set; } = 20;

        /// <summary>
        /// 切句最小字数。LLM 流必须累积到该长度且命中标点才会切出一句喂给 TTS。
        /// 太小会让 TTS 频繁启动, 太大会拖高首音延迟。
        /// </summary>
        public int MinSentenceChars { get; set; } = 8;

        /// <summary>
        /// 单次 LLM 推理的 token 上限 (覆盖 GemmaLLMOptions.MaxTokens)。null 表示沿用 LLM 服务默认。
        /// </summary>
        public int? MaxResponseTokens { get; set; }

        /// <summary>
        /// 指定 TTS 音色。null 表示用 TTS 服务的默认音色。
        /// </summary>
        public string? TtsVoice { get; set; }

        /// <summary>
        /// 指定 TTS 语速倍率。null 表示用 TTS 服务的默认语速；1.0 表示正常语速。
        /// 目前只有支持 <see cref="Core.Services.TTSService.Models.TtsSynthesisOptions.Speed"/> 的 TTS 实现会生效。
        /// </summary>
        public float? TtsSpeed { get; set; }

        /// <summary>
        /// 播放队列排空的额外缓冲毫秒数, 加在估算播放时长之后以确保最后一帧出声。
        /// </summary>
        public int PlaybackTailBufferMs { get; set; } = 250;

        /// <summary>
        /// 播放预缓冲 (抖动缓冲) 毫秒数。每轮回复开播前先攒够这么长的音频再一次性灌入播放队列,
        /// 给播放一个领先量; 之后逐块直通。可吸收合成偶发跟不上 1x 播放速度 (尤其是句与句之间的
        /// 首包等待) 造成的卡顿/停顿。代价是每轮首音延迟增加约这个毫秒数。
        /// 0 表示关闭预缓冲 (合成出一块立即播一块, 旧版行为)。卡顿明显时可调大到 300~500。
        /// </summary>
        public int PlaybackPrerollMs { get; set; } = 250;

        /// <summary>
        /// 对话行为配置。
        /// </summary>
        public VoiceAssistantConversationOptions Conversation { get; set; } = new();

        /// <summary>
        /// 唤醒词门控配置。
        /// </summary>
        public VoiceAssistantWakeupOptions Wakeup { get; set; } = new();

        /// <summary>本人说话人验证配置。</summary>
        public VoiceAssistantSpeakerOptions Speaker { get; set; } = new();

        /// <summary>
        /// Idle/连续对话态监听确认配置。
        /// </summary>
        public VoiceAssistantListeningOptions Listening { get; set; } = new();

        /// <summary>
        /// 可打断回复 (barge-in) 配置。
        /// </summary>
        public VoiceAssistantBargeInOptions BargeIn { get; set; } = new();

        /// <summary>
        /// 结构化工具调用配置。仅在安装了工具能力时生效。
        /// </summary>
        public VoiceAssistantToolOptions Tools { get; set; } = new();

        internal VoiceAssistantOptions Clone()
        {
            // The DTD model settings carry a SessionOptionsFactory delegate that System.Text.Json
            // cannot serialize; it is skipped by the shared clone contract and re-attached by
            // reference here. A factory is a borrowed collaborator, so sharing it is correct, and
            // dropping it would silently downgrade a caller-supplied ONNX session to the default.
            string json = JsonSerializer.Serialize(this, AssistantOptionsCloneSerializer.Json);
            VoiceAssistantOptions clone = JsonSerializer.Deserialize<VoiceAssistantOptions>(
                json,
                AssistantOptionsCloneSerializer.Json)
                ?? throw new InvalidOperationException("VoiceAssistantOptions could not be cloned.");
            if (BargeIn?.Dtd?.Model?.SessionOptionsFactory is { } sessionOptionsFactory &&
                clone.BargeIn?.Dtd?.Model is { } clonedModel)
            {
                clonedModel.SessionOptionsFactory = sessionOptionsFactory;
            }

            return clone;
        }
    }
}
