using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Wakeup.OpenWakeWord.Models
{
    /// <summary>
    /// OpenWakeWord 唤醒服务的配置选项。
    /// </summary>
    public class OpenWakeWordOptions : OnnxExecutionOptions
    {
        private static readonly string DefaultModelDir =
            Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "wakewords");

        /// <summary>
        /// 录音采样率 (Hz)。OpenWakeWord 仅支持 16000Hz。
        /// </summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>
        /// 录音声道数。默认 1 (单声道)。
        /// </summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// 采样格式。仅支持 Float32 (内部会按 PCM int16 振幅做规约)。
        /// </summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// Mel 频谱图特征提取模型 (melspectrogram.onnx) 路径。
        /// </summary>
        public string MelSpectrogramModelPath { get; set; } =
            Path.Combine(DefaultModelDir, "melspectrogram.onnx");

        /// <summary>
        /// 音频嵌入特征模型 (embedding_model.onnx, Google speech_embedding) 路径。
        /// </summary>
        public string EmbeddingModelPath { get; set; } =
            Path.Combine(DefaultModelDir, "embedding_model.onnx");

        /// <summary>
        /// 启用的唤醒词模型列表。至少需要 1 个。
        /// </summary>
        public IList<WakeWordModelDefinition> WakeWords { get; } = new List<WakeWordModelDefinition>();

        /// <summary>
        /// 添加一个唤醒词模型 (链式调用)。
        /// </summary>
        public OpenWakeWordOptions AddWakeWord(string keyword, string modelPath, float threshold = 0.5f, int debounceMs = 1000)
        {
            WakeWords.Add(new WakeWordModelDefinition
            {
                Keyword = keyword,
                ModelPath = modelPath,
                Threshold = threshold,
                DebounceMs = debounceMs,
            });
            return this;
        }
    }
}
