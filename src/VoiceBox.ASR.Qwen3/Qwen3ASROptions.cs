using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.Qwen3
{
    /// <summary>
    /// Configuration for <see cref="Qwen3ASRService"/>.
    /// 模型加载、推理调度与流式重转写参数均通过本类暴露。
    /// </summary>
    public sealed class Qwen3ASROptions : OnnxExecutionOptions
    {
        /// <summary>
        /// ONNX 模型目录。需包含 metadata.json / encoder 或 (conv_embed + audio_stack)
        /// 模型文件、prefill+decode 解码器、tokenizer 文件以及 preprocessor_config.json。
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>解码时最多生成的 token 数（不含 prompt）。</summary>
        public int MaxNewTokens { get; set; } = 512;

        /// <summary>
        /// 当用户在 StartSession 时未显式指定 language 时是否让模型自动检测。
        /// 设为非空字符串可在 prompt 中强制 <c>language {Lang}&lt;asr_text&gt;</c>。
        /// </summary>
        public string? ForceLanguage { get; set; }

        /// <summary>
        /// 流式滚动重转写：未稳定的前 N 个分块不会把累计文本作为 prefix，避免早期抖动。
        /// 对应 Python ASRStreamingState.unfixed_chunk_num。
        /// </summary>
        public int UnfixedChunkNum { get; set; } = 1;

        /// <summary>
        /// 流式滚动重转写：稳定阶段把累计文本末尾的 K 个 token 视作不稳定并回滚，
        /// 仅取前缀作为 prefix。对应 Python ASRStreamingState.unfixed_token_num。
        /// </summary>
        public int UnfixedTokenNum { get; set; } = 5;

        /// <summary>流式滚动窗口长度（秒）。每凑齐一个窗口便触发一次完整转写。</summary>
        public double ChunkSizeSec { get; set; } = 3.0;

        /// <summary>单会话音频块缓冲上限（每次 FeedAudio 即一个块），溢出时丢帧并打印告警。</summary>
        public int MaxBufferChunks { get; set; } = 128;

        /// <summary>同时存活的最大会话数。</summary>
        public int MaxConcurrentSessions { get; set; } = 8;

        /// <summary>
        /// 同时执行 ONNX 推理的最大并发度。必须为 1。Qwen3-ASR 的 encoder、
        /// decoder、KV cache 和 scratch buffer 在一个 service 实例内共享，
        /// 不支持并行写入。
        /// <para>
        /// 如需扩展吞吐，应使用多个 service 实例（每个实例拥有独立的 ONNX runtime）。
        /// </para>
        /// </summary>
        public int MaxConcurrentInferences { get; set; } = 1;

        /// <summary>FinishSession 时等待 pump 排空音频的超时（秒）。0 表示无限等待。</summary>
        public int FinishDrainTimeoutSec { get; set; } = 30;

        /// <summary>system message 的默认 context（可被 StartSession 入参覆盖）。</summary>
        public string DefaultContext { get; set; } = string.Empty;

        /// <summary>
        /// 是否在 CUDA EP 下让 decode 使用 IOBinding + GPU 驻留 KV cache。
        /// 仅在 (a) <see cref="OnnxExecutionOptions.ExecutionProvider"/> 为 Cuda 且
        /// (b) 模型 <c>decode_cache_mode</c> 非 <c>delta</c> (即 cache 输入/输出形状一致)
        /// 时才会生效——此时 decoder 用双缓冲 GPU 张量池避免每步 ~190 MB 的 H2D 拷贝。
        /// Delta 模式下此开关被忽略 (delta scatter 必须在 GPU 完成, 当前 ORT C# 暴露的
        /// 原语无法完成, 强行启用反而会引入 GPU→CPU readback)。CPU/DML EP 也忽略。
        /// 默认 true; 调试 IOBinding 兼容性问题时可临时关闭以走回 Dictionary 路径。
        /// </summary>
        public bool EnableIoBinding { get; set; } = true;
    }
}
