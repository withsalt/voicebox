using VoiceBox.Core.Onnx;

namespace VoiceBox.TTS.VoxCPM
{
    /// <summary>
    /// VoxCPM-0.5B ONNX 服务配置。
    ///
    /// 模型目录布局 (由 scripts/voxcpm-tts-onnx-export/export.py 生成):
    ///   {ModelPath}/lm.onnx                    统一 prefill/decode LM (base+residual, KV-cache)
    ///   {ModelPath}/feat_encoder.onnx          latent patch → LM 嵌入
    ///   {ModelPath}/cfm_decode.onnx            整段 10 步 Euler 流匹配 (噪声/cfg 为输入)
    ///   {ModelPath}/audio_vae_decoder.onnx     latents → 16kHz 波形 (恒 fp32)
    ///   {ModelPath}/audio_vae_encoder.onnx     波形 → latents, 音色克隆用 (恒 fp32)
    ///   {ModelPath}/embeddings/text_embedding.bin + multichar_zh.json
    ///   {ModelPath}/tokenizer/tokenizer.json
    ///   {ModelPath}/config.json / metadata.json
    ///
    /// 架构: 文本 → BPE(+中文拆字) → prefill → 循环 { CFM 扩散出 1 patch (80ms)
    ///       → stop 头判停 → LM 单步 } → AudioVAE 解码。流式每步发 80ms。
    ///
    /// 生成相关参数为可空覆盖: 为 null 时使用包内 config.json 的 generation 默认值;
    /// 单次合成可经 <see cref="Core.Services.TTSService.Models.TtsSynthesisOptions"/> 再覆盖。
    /// </summary>
    public class VoxCPMTTSOptions : OnnxExecutionOptions
    {
        public override int ThreadsNum { get; set; } = Math.Clamp(Environment.ProcessorCount / 2, 2, 8);

        /// <summary>VoxCPM 模型包根目录。</summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>音色克隆声音列表。除此之外始终有内置 "default" (模型自由发挥音色)。</summary>
        public List<VoxCpmCloneVoice> CloneVoices { get; set; } = [];

        /// <summary>默认音色。为空取 "default" (无 prompt)。</summary>
        public string? DefaultVoice { get; set; }

        // ---- 生成参数 (null = 用包内 config.json 默认) ----

        /// <summary>Classifier-free guidance 强度。包默认 2.0。</summary>
        public float? CfgValue { get; set; }

        /// <summary>最小生成 patch 数 (每 patch 80ms), 防过短音频。包默认 2。</summary>
        public int? MinLen { get; set; }

        /// <summary>
        /// 单次合成最多生成的 patch 数上限 (还受 文本token数*6+10 与模型 4096 位置约束)。
        /// 是服务级护栏: 单次调用的 TtsSynthesisOptions.MaxNewFrames 只能收紧它。null 用包 max_length。
        /// </summary>
        public int? MaxLen { get; set; }

        /// <summary>
        /// 默认随机种子 (CFM 初始噪声)。null 表示每次随机 —— 同文本两次合成韵律会不同。
        /// 设为固定值时 (text, voice, seed) 决定输出。
        /// </summary>
        public int? DefaultSeed { get; set; }

    }

    /// <summary>
    /// 音色克隆声音: 一段 prompt 音频 + 对应文本。首次使用时经 AudioVAE 编码并缓存。
    /// </summary>
    public sealed class VoxCpmCloneVoice
    {
        /// <summary>音色名 (经 TtsSynthesisOptions.Voice 选择)。</summary>
        public required string Name { get; init; }

        /// <summary>prompt 音频路径 (wav; 任意采样率, 自动重采样到 16k 单声道)。</summary>
        public required string PromptWavPath { get; init; }

        /// <summary>prompt 音频的文本内容 (必须与音频一致, 影响克隆质量)。</summary>
        public required string PromptText { get; init; }
    }
}
