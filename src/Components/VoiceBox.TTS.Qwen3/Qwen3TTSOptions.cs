using VoiceBox.Core.Onnx;

namespace VoiceBox.TTS.Qwen3
{
    /// <summary>
    /// Qwen3-TTS (12Hz CustomVoice) ONNX 服务配置。
    ///
    /// 模型目录布局 (由 scripts/qwen3-tts-onnx-export/export.py 生成):
    ///   {ModelPath}/talker.onnx                统一 prefill/decode talker (KV-cache)
    ///   {ModelPath}/code_predictor.onnx        子talker 单步图 (generation_steps → head gather)
    ///   {ModelPath}/codec_decoder.onnx         codec → 24kHz 波形 (恒 fp32)
    ///   {ModelPath}/embeddings/*.bin + config.json   预投影文本/codec 嵌入 + 全量模型/生成配置
    ///   {ModelPath}/tokenizer/{vocab.json,merges.txt,tokenizer_config.json}   Qwen BBPE
    ///   {ModelPath}/metadata.json
    ///
    /// 架构: 文本 → Qwen 分词 → 拼 prompt 嵌入 → talker 自回归 (每帧出第0组 codec)
    ///       → code_predictor 每帧再自回归 15 步补满 16 组 → codec 解码 → 24kHz 波形。
    ///
    /// 生成相关参数为可空覆盖: 为 null 时使用包内 config.json 的 generation 默认值;
    /// 单次合成可经 <see cref="Core.Services.TTSService.Models.TtsSynthesisOptions"/> 再覆盖。
    /// </summary>
    public class Qwen3TTSOptions : OnnxExecutionOptions
    {
        public override int ThreadsNum { get; set; } = Math.Clamp(Environment.ProcessorCount / 2, 2, 8);

        /// <summary>Qwen3-TTS 模型包根目录。</summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>默认音色 (speaker)。为空时取包内 speakerId 的第一个。</summary>
        public string? DefaultVoice { get; set; }

        /// <summary>默认语言 ("auto"/"chinese"/"english"/...)。为空默认 "auto"。</summary>
        public string? DefaultLanguage { get; set; } = "auto";

        // ---- 生成参数 (null = 用包内 config.json 默认) ----

        /// <summary>是否采样。false 为贪心 (确定性, 与 Python greedy 逐 token 一致)。</summary>
        public bool? DoSample { get; set; }

        /// <summary>talker 第0组采样温度。</summary>
        public float? Temperature { get; set; }

        /// <summary>talker top-k。</summary>
        public int? TopK { get; set; }

        /// <summary>talker top-p。</summary>
        public float? TopP { get; set; }

        /// <summary>talker 重复惩罚 (作用于已生成的第0组 token 序列)。默认 1.05。</summary>
        public float? RepetitionPenalty { get; set; }

        /// <summary>code_predictor (子talker) 是否采样。</summary>
        public bool? SubtalkerDoSample { get; set; }

        /// <summary>子talker 采样温度 (运行时传入融合图)。</summary>
        public float? SubtalkerTemperature { get; set; }

        /// <summary>
        /// 子talker top-k。<b>注意</b>: 融合版 code_predictor.onnx 在导出时已烘焙 top-k (取
        /// generation_config 的 subtalker_top_k, 默认 50) 以保持静态图; 运行时设置此值<b>不生效</b>。
        /// 需要不同 top-k 请用对应参数重新导出。
        /// </summary>
        public int? SubtalkerTopK { get; set; }

        /// <summary>
        /// 子talker top-p。<b>注意</b>: 融合图未实现 nucleus 采样 (默认 top_p=1.0 即不裁剪, 与参考一致);
        /// 设置 &lt;1 的值<b>不生效</b>。
        /// </summary>
        public float? SubtalkerTopP { get; set; }

        /// <summary>单次合成最多生成的音频帧数 (12.5Hz, 即每帧 80ms)。null 用包默认。</summary>
        public int? MaxNewFrames { get; set; }

        /// <summary>
        /// 默认随机种子。null 表示每次使用全局递增/随机种子。设为固定值时 (text, voice, seed) 决定输出。
        /// </summary>
        public int? DefaultSeed { get; set; } = 23;

        // ---- 流式 ----

        /// <summary>
        /// 流式各块累计多少新帧后解码并发出 (末值循环)。小首块降低首包延迟; codec 解码与生成在
        /// 后台线程重叠, 故较大的后续块减少左上下文冗余、RTF 更低。默认 [1,4,8,16,32,48]: 首块 1 帧
        /// (≈0.08s) 降低首包延迟; 后续渐增。首块越小首包越低但越易欠载 (配合播放预缓冲)。
        /// </summary>
        public int[] StreamChunkSchedule { get; set; } = [2, 4, 8, 16, 16, 16];

        /// <summary>codec 分块解码的左上下文帧数 (与 Python chunked_decode 的 left_context 一致)。默认 25。</summary>
        public int StreamLeftContextFrames { get; set; } = 25;

        /// <summary>相邻 codec 块之间的交叉淡入秒数 (0 表示不淡入)。默认 0。</summary>
        public float StreamFadeSeconds { get; set; } = 0f;
    }
}
