using System.Collections.Generic;

namespace VoiceBox.ASR.DoubaoOnline
{
    /// <summary>
    /// <see cref="DoubaoOnlineASRService"/> 的配置。涵盖鉴权、接入模式、音频参数、
    /// 识别行为与连接调度等。默认值面向「豆包流式语音识别模型 2.0 + 双向流式 + 新版控制台」。
    /// </summary>
    public sealed class DoubaoOnlineASROptions
    {
        // ---- 鉴权 (新版控制台 ApiKey) ----

        /// <summary>
        /// 新版控制台 API Key，对应请求头 <c>X-Api-Key</c>。
        /// 新版控制台只需设置该字段即可完成鉴权。
        /// </summary>
        public string? ApiKey { get; set; }

        /// <summary>
        /// 资源信息 ID，对应请求头 <c>X-Api-Resource-Id</c>。默认豆包 2.0 小时版。
        /// 可用取值见 <see cref="SeedAsrResourceIds"/>。
        /// </summary>
        public string ResourceId { get; set; } = SeedAsrResourceIds.SeedAsrDuration;

        // ---- 接入点 ----

        /// <summary>WebSocket 服务主机 (不含协议与路径)。</summary>
        public string Host { get; set; } = "openspeech.bytedance.com";

        /// <summary>接入模式，决定 WebSocket 路径。默认双向流式模式。</summary>
        public SeedAsrStreamMode StreamMode { get; set; } = SeedAsrStreamMode.Bidirectional;

        /// <summary>是否使用 <c>wss</c> (TLS)。默认 true；仅调试可关。</summary>
        public bool UseTls { get; set; } = true;

        /// <summary>附加到握手请求上的自定义请求头 (例如灰度标记)。可为空。</summary>
        public IDictionary<string, string>? ExtraHeaders { get; set; }

        // ---- 音频参数 ----

        /// <summary>采样率，目前接口仅支持 16000。</summary>
        public int SampleRate { get; set; } = 16000;

        /// <summary>声道数，1(mono)/2(stereo)，默认 1。</summary>
        public int Channels { get; set; } = 1;

        /// <summary>
        /// 单个音频包的时长 (毫秒)。文档建议 100~200ms。
        /// </summary>
        public int PackageDurationMs { get; set; } = 200;

        // ---- 识别行为 (映射到 request 字段) ----

        /// <summary>模型名称，目前只有 <c>bigmodel</c>。</summary>
        public string ModelName { get; set; } = "bigmodel";

        /// <summary>启用文本规范化 (ITN)，默认 true。</summary>
        public bool EnableItn { get; set; } = true;

        /// <summary>启用标点，默认 true。</summary>
        public bool EnablePunc { get; set; } = true;

        /// <summary>启用语义顺滑 (DDC)，默认 false。</summary>
        public bool EnableDdc { get; set; }

        /// <summary>输出语音停顿/分句/分词信息 (show_utterances)，默认 true 以便区分已确定/未确定文本。</summary>
        public bool ShowUtterances { get; set; } = true;

        /// <summary>
        /// 结果返回方式：<c>full</c> 全量返回 (默认)，<c>single</c> 增量返回 (不含之前分句)。
        /// </summary>
        public string ResultType { get; set; } = "full";

        /// <summary>强制判停时间 (ms)，默认服务端 800，最小 200。为 0 表示不设置该字段。</summary>
        public int EndWindowSize { get; set; }

        /// <summary>热词直传，形如 <c>{"hotwords":[{"word":"热词1"}]}</c>。为空则不发送。</summary>
        public string? HotWords { get; set; }

        /// <summary>用户标识 (user.uid)，可用于服务端日志过滤。</summary>
        public string? Uid { get; set; }

        // ---- 连接调度 ----

        /// <summary>同时存活的最大会话数 (每会话一条独立 WebSocket 连接)。</summary>
        public int MaxConcurrentSessions { get; set; } = 16;

        /// <summary>WebSocket 建连超时 (秒)。</summary>
        public int ConnectTimeoutSec { get; set; } = 10;

        /// <summary>FinishSession 等待最终结果的超时 (秒)。0 表示无限等待。</summary>
        public int FinishTimeoutSec { get; set; } = 15;

        /// <summary>是否对上行音频做 Gzip 压缩。默认 true (与官方 Demo 一致)。</summary>
        public bool CompressAudio { get; set; } = true;

        /// <summary>
        /// 是否将上行发送节流到实时速率 (1x)。对麦克风实时流无副作用 (本就实时到达)；
        /// 对整段音频回灌可避免瞬间发包过快触发服务端限流。默认 true。
        /// </summary>
        public bool ThrottleSendToRealtime { get; set; } = true;

        /// <summary>
        /// 单会话待发送音频的最大时长。容量按实际采样数记账，达到上限会明确终止
        /// 会话，绝不会静默丢弃句首。默认 8 秒。
        /// </summary>
        public int MaxBufferedAudioMs { get; set; } = 8000;

        /// <summary>
        /// 兼容旧配置的缓冲块上限。实现会按 <see cref="PackageDurationMs"/> 换算为时长，
        /// 并与 <see cref="MaxBufferedAudioMs"/> 取较小值；实际记账始终使用采样数。
        /// </summary>
        public int MaxBufferChunks { get; set; } = 256;
    }
}
