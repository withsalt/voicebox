using System.Text.Json;
using VoiceBox.Assistant.Composition.Configuration;

namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 语音助手运行参数。
    /// </summary>
    public sealed class VoiceAssistantOptions
    {
        /// <summary>
        /// 切句最小字数。LLM 流必须累积到该长度且命中标点才会切出一句喂给 TTS。
        /// 太小会让 TTS 频繁启动, 太大会拖高首音延迟。
        /// </summary>
        public int MinSentenceChars { get; set; } = 8;

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

        internal VoiceAssistantOptions Clone()
        {
            // The DTD model settings carry a SessionOptionsFactory delegate that System.Text.Json
            // cannot serialize; it is skipped by the shared clone contract and re-attached by
            // reference here. A factory is a borrowed collaborator, so sharing it is correct, and
            // dropping it would silently downgrade a caller-supplied ONNX session to the default.
            string json = JsonSerializer.Serialize(this, AssistantOptionsCloneSerializer.Json);
            VoiceAssistantOptions clone = JsonSerializer.Deserialize<VoiceAssistantOptions>(json, AssistantOptionsCloneSerializer.Json)
                ?? throw new InvalidOperationException("VoiceAssistantOptions could not be cloned.");
            if (BargeIn?.Dtd?.Model?.SessionOptionsFactory is { } sessionOptionsFactory && clone.BargeIn?.Dtd?.Model is { } clonedModel)
            {
                clonedModel.SessionOptionsFactory = sessionOptionsFactory;
            }

            return clone;
        }
    }
}
