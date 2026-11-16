using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.VAD.FireRed.Models
{
    public class FireRedVadOptions : OnnxExecutionOptions
    {
        /// <summary>
        /// ONNX Runtime intra-op 线程数。VAD 模型较小，且每帧处理时间远小于 10 ms，过多线程反而可能增加调度开销和回调竞争。
        /// </summary>
        public override int ThreadsNum { get; set; } = 1;

        /// <summary>
        /// 录音采样率 (Hz)。FireRedVAD 仅支持 16000Hz。
        /// </summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>
        /// 录音声道数。
        /// </summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// 采样格式。
        /// </summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// 平滑滑动窗口大小（帧）。每帧 10ms。
        /// </summary>
        public int SmoothWindowSize { get; set; } = 5;

        /// <summary>
        /// 判定为语音的概率阈值（已平滑）。
        /// </summary>
        public float SpeechThreshold { get; set; } = 0.55f;

        /// <summary>
        /// 检测到语音起点时向前扩展的帧数（用于回溯一段前置音频）。每帧 10ms。
        /// </summary>
        public int PadStartFrame { get; set; } = 5;

        /// <summary>
        /// 进入 SPEECH 状态所需的连续语音帧数。每帧 10ms。
        /// </summary>
        public int MinSpeechFrame { get; set; } = 8;

        /// <summary>
        /// 单段语音最长帧数。超过则强制结束当前语音段。每帧 10ms。
        /// </summary>
        public int MaxSpeechFrame { get; set; } = 3000;

        /// <summary>
        /// 判定语音结束所需的连续静音帧数。每帧 10ms。
        /// </summary>
        public int MinSilenceFrame { get; set; } = 60;

        /// <summary>
        /// 检测到语音前要保留的音频时长 (毫秒)。语音开始事件触发时，会一并返回这部分前置音频。
        /// 设为 0 可关闭。默认值: 300。
        /// </summary>
        public int PreSpeechPaddingMs { get; set; } = 300;

        /// <summary>
        /// VAD 模型路径（fireredvad_stream_vad_with_cache.onnx）。
        /// </summary>
        public string VADModelPath { get; set; } = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "vad", "fireredvad", "fireredvad_stream_vad_with_cache.onnx");

        /// <summary>
        /// Kaldi CMVN 统计文件路径（cmvn.ark）。
        /// </summary>
        public string CMVNPath { get; set; } = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "vad", "fireredvad", "cmvn.ark");
    }
}
