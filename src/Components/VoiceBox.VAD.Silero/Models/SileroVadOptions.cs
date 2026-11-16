using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.VAD.Silero.Models
{
    public class SileroVadOptions : OnnxExecutionOptions
    {
        /// <summary>
        /// 录音采样率 (Hz)。
        /// 默认值: 16000。
        /// </summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>
        /// 录音声道数。
        /// 默认值: 1 (单声道)。
        /// </summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// 采样格式
        /// </summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// The probability threshold for speech detection.
        /// </summary>
        public float Threshold { get; set; } = 0.5f;

        /// <summary>
        /// Minimum duration of silence in milliseconds to consider the end of a speech segment.
        /// </summary>
        public int MinSilenceDurationMs { get; set; } = 300;

        /// <summary>
        /// 单段语音的最长时长 (毫秒)。当 VAD 持续把输入判为语音 (例如电视/音乐/嘈杂环境/常开麦克风)
        /// 时, 若没有上限, 内部语音缓冲会随帧无限增长直至 OutOfMemoryException。到达此上限会强制结束
        /// 当前语音段 (触发 <c>OnSpeechEnded</c>), 下一帧语音重新开始新段 —— 与 FireRedVAD 的
        /// MaxSpeechFrame 行为一致。设为 0 可关闭该保护 (不推荐: 长时间运行可能 OOM)。
        /// 默认值: 30000 (30 秒)。正常时长的语音 (远小于该值) 行为完全不受影响。
        /// </summary>
        public int MaxSpeechDurationMs { get; set; } = 30000;

        /// <summary>
        /// Number of milliseconds of audio to keep from before speech was detected.
        /// </summary>
        public int PreSpeechPaddingMs { get; set; } = 300;

        /// <summary>
        /// VAD model path.
        /// </summary>
        public string VADModelPath { get; set; } = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "vad", "silero", "silero_vad.onnx");

        /// <summary>
        /// Number of threads for ONNX Runtime. Silero VAD is a small model and runs on short streaming frames, so a small ORT thread pool can help avoid oversubscribing CPU cores when combined with AEC/VAD in live pipelines.
        /// </summary>
        public override int ThreadsNum { get; set; } = 1;
    }
}
