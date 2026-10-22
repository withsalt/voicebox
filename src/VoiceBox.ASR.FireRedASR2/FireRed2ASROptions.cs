using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.FireRedASR2
{
    public class FireRed2ASROptions : OnnxExecutionOptions
    {
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// 单会话音频缓冲区最大块数（每块对应一次 FeedAudio 调用）。
        /// 缓冲区满时按 DropOldest 策略丢弃最旧的一块（之前是 DropWrite，但配合
        /// .NET Channels 的 TryWrite 返回语义会造成 ArrayPool 泄漏与丢帧统计失效）。
        /// 默认 64 块在正常压力下不会触发丢弃。
        /// </summary>
        public int MaxBufferChunks { get; set; } = 64;

        /// <summary>
        /// 同时允许活动的最大会话数；超过则 StartSessionAsync 抛出 InvalidOperationException。
        /// </summary>
        public int MaxConcurrentSessions { get; set; } = 16;

        /// <summary>
        /// 同时进行 ONNX 推理（GetResult）的最大并发度。
        /// 默认与 ThreadsNum 解耦：识别本身较重，过高的并发会让 CPU 与内存竞争反而变慢。
        /// </summary>
        public int MaxConcurrentRecognitions { get; set; } = 1;

        /// <summary>
        /// FinishSessionAsync 等待 pump 排空音频的超时（秒）。0 表示无限等待。
        /// </summary>
        public int FinishDrainTimeoutSec { get; set; } = 30;
    }
}
