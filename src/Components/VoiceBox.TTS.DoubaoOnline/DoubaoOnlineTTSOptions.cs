using System.Collections.Generic;

namespace VoiceBox.TTS.DoubaoOnline
{
    /// <summary>
    /// <see cref="DoubaoOnlineTTSService"/> 的配置。涵盖鉴权、接入点、音频参数、合成行为与连接调度。
    /// 默认值面向「豆包语音合成大模型 2.0 (seed-tts-2.0) + 双向流式 + 新版控制台」。
    /// </summary>
    /// <remarks>
    /// 说明：本服务经 Core 的 <c>IStreamTTSService</c> 契约输出 <c>float[]</c> PCM，故内部固定请求
    /// <c>pcm</c> (s16le) 音频格式；<see cref="SampleRate"/> 决定输出采样率。若需 mp3/ogg 等编码格式，
    /// 需在上层自行解码，不在本服务职责内。
    /// </remarks>
    public sealed class DoubaoOnlineTTSOptions
    {
        // ---- 鉴权 (新版控制台 ApiKey) ----

        /// <summary>
        /// 新版控制台 API Key，对应请求头 <c>X-Api-Key</c>。新版控制台只需设置该字段即可完成鉴权。
        /// </summary>
        public string? ApiKey { get; set; }

        /// <summary>
        /// 资源信息 ID，对应请求头 <c>X-Api-Resource-Id</c>。默认豆包语音合成大模型 2.0。
        /// 可用取值见 <see cref="SeedTtsResourceIds"/> (复刻音色用 <see cref="SeedTtsResourceIds.SeedIcl20"/>)。
        /// </summary>
        public string ResourceId { get; set; } = SeedTtsResourceIds.SeedTts20;

        /// <summary>是否请求返回计费字符数 (请求头 <c>X-Control-Require-Usage-Tokens-Return: *</c>)。默认 false。</summary>
        public bool ReturnUsage { get; set; }

        // ---- 接入点 ----

        /// <summary>WebSocket 服务主机 (不含协议与路径)。</summary>
        public string Host { get; set; } = "openspeech.bytedance.com";

        /// <summary>WebSocket 请求路径。默认双向流式合成接口。</summary>
        public string Path { get; set; } = "/api/v3/tts/bidirection";

        /// <summary>是否使用 <c>wss</c> (TLS)。默认 true；仅调试可关。</summary>
        public bool UseTls { get; set; } = true;

        /// <summary>附加到握手请求上的自定义请求头 (例如灰度标记)。可为空。</summary>
        public IDictionary<string, string>? ExtraHeaders { get; set; }

        // ---- 合成参数 (映射到 StartSession 的 req_params) ----

        /// <summary>
        /// 默认音色 ID (speaker)，例如 <c>zh_female_tianmeitaozi_mars_bigtts</c>。
        /// 可被单次合成的 <c>TtsSynthesisOptions.Voice</c> 覆盖。可从「控制台 &gt; 音色库」或
        /// <see cref="SpeakerCatalog.VolcSpeakerCatalogClient"/> 获取。
        /// </summary>
        public string? DefaultVoice { get; set; }

        /// <summary>
        /// 具体模型版本 (req_params.model)，仅在使用复刻音色时需要，例如 <c>seed-tts-2.0-standard</c>。
        /// 为空则不发送该字段 (使用大模型 2.0 音色时留空)。
        /// </summary>
        public string? Model { get; set; }

        /// <summary>
        /// 供 <c>ListVoices()</c> 返回的已知音色清单 (name/description/language)。
        /// 在线服务本身不内置音色表；此列表仅用于本地展示/选择，不影响实际合成。可为空。
        /// </summary>
        public IReadOnlyList<SeedTtsVoiceInfo>? KnownVoices { get; set; }

        // ---- 音频参数 (audio_params) ----

        /// <summary>
        /// 音频采样率 (Hz)。可选值：8000/16000/22050/24000/32000/44100/48000。默认 24000。
        /// </summary>
        public int SampleRate { get; set; } = 24000;

        /// <summary>输出声道数。当前接口输出单声道，固定 1。</summary>
        public int ChannelCount { get; set; } = 1;

        /// <summary>语速，取值范围 [-50,100]，100=2.0 倍速，-50=0.5 倍速。默认 0 (原速)。</summary>
        public int SpeechRate { get; set; }

        /// <summary>音量，取值范围 [-50,100]，100=2.0 倍音量，-50=0.5 倍音量。默认 0 (原音量)。</summary>
        public int LoudnessRate { get; set; }

        /// <summary>音调 (post_process.pitch)，取值范围 [-12,12]。默认 0 (不设置该字段)。</summary>
        public int Pitch { get; set; }

        /// <summary>是否开启字幕 (字级别时间戳)。仅豆包 2.0 支持，且目前仅中英文。默认 false。</summary>
        public bool EnableSubtitle { get; set; }

        // ---- additions (文本处理开关) ----

        /// <summary>是否过滤 Markdown 语法 (disable_markdown_filter)。默认 false (不过滤)。</summary>
        public bool DisableMarkdownFilter { get; set; }

        /// <summary>是否过滤 Emoji (disable_emoji_filter)。默认 false (不过滤)。</summary>
        public bool DisableEmojiFilter { get; set; }

        /// <summary>显式指定朗读语种 (explicit_language)，例如 <c>zh-cn</c>/<c>en</c>/<c>ja</c>。为空不设置。</summary>
        public string? ExplicitLanguage { get; set; }

        /// <summary>指定方言 (explicit_dialect)。需配合支持方言的音色。为空不设置。</summary>
        public string? ExplicitDialect { get; set; }

        // ---- 多轮 / 语音指令 ----

        /// <summary>
        /// 语音指令 (context_texts)，例如 <c>["你可以用特别痛心的语气说话吗?"]</c>。
        /// 仅豆包 2.0 音色支持；复刻音色暂不支持。该文本不参与计费。为空不发送。
        /// </summary>
        public IReadOnlyList<string>? ContextTexts { get; set; }

        /// <summary>
        /// 多轮会话 ID (section_id)，用于关联同一上下文的多次串行合成 (服务端保存历史)。
        /// 建议一通对话用一个 UUID。为空不发送。
        /// </summary>
        public string? SectionId { get; set; }

        // ---- 连接调度 ----

        /// <summary>同时存活的最大会话数 (每会话一条独立 WebSocket 连接)。</summary>
        public int MaxConcurrentSessions { get; set; } = 16;

        /// <summary>WebSocket 建连超时 (秒)。</summary>
        public int ConnectTimeoutSec { get; set; } = 10;

        /// <summary>
        /// 单次合成等待服务端收尾 (SessionFinished) 的超时 (秒)。0 表示无限等待。
        /// </summary>
        public int SessionTimeoutSec { get; set; } = 10;

        /// <summary>是否对上行 JSON payload 做 Gzip 压缩。默认 false (与官方 TTS demo 一致)。</summary>
        public bool CompressPayload { get; set; }
    }

    /// <summary>
    /// 本地已知音色信息，供 <c>ListVoices()</c> 返回。与在线接口无关，仅用于本地展示/选择。
    /// </summary>
    public sealed class SeedTtsVoiceInfo
    {
        /// <summary>音色 ID (speaker)，用于实际合成。</summary>
        public required string Name { get; init; }

        /// <summary>展示名 / 描述。</summary>
        public string? Description { get; init; }

        /// <summary>语言标识 (例如 zh-cn)。</summary>
        public string? Language { get; init; }
    }
}
