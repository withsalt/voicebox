using VoiceBox.Core.Onnx;

namespace VoiceBox.TTS.Kokoro
{
    public class KokoroTTSOptions : OnnxExecutionOptions
    {
        /// <summary>
        /// Kokoro 的 CPU 推理以较大的矩阵/卷积算子为主，默认多给一些 intra-op 线程以降低首包等待。
        /// </summary>
        public override int ThreadsNum { get; set; } = Math.Clamp(Environment.ProcessorCount / 2, 2, 8);

        /// <summary>
        /// ONNX 模型文件路径 (如 kokoro-v1.0.onnx)。
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// 音色文件路径 (如 voices-v1.0.bin，NumPy .npy/.npz 格式)。
        /// </summary>
        public string VoicesPath { get; set; } = string.Empty;

        /// <summary>
        /// 词汇表配置文件路径 (config.json)。为空时使用内置默认词汇表。
        /// </summary>
        public string? VocabConfigPath { get; set; }

        /// <summary>
        /// 默认音色名称。为空时使用第一个可用音色。
        /// </summary>
        public string? DefaultVoice { get; set; }

        /// <summary>
        /// 默认语速 (0.5 ~ 2.0)。
        /// </summary>
        public float DefaultSpeed { get; set; } = 1.0f;

        /// <summary>
        /// 是否使用 WeTextProcessing.NET 对输入文本进行口语化归一。默认启用。
        /// </summary>
        public bool EnableTextNormalization { get; set; } = true;

        /// <summary>
        /// 同时进行合成的最大并发数。
        /// </summary>
        public int MaxConcurrentSyntheses { get; set; } = 1;

        /// <summary>
        /// 流式合成的首批 phoneme 预算 (单位: 字符)。
        /// 较小的值会减少首块等待；较大的值会保留更多上下文。
        /// 取值范围建议 50 ~ 510；&lt;= 0 表示不启用短首批策略。
        /// </summary>
        public int FirstBatchPhonemeBudget { get; set; } = 100;
    }
}
