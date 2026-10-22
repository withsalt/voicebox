using VoiceBox.Core.Onnx;

namespace VoiceBox.TTS.MossNano
{
    /// <summary>
    /// MOSS-TTS-Nano (ONNX) 服务配置。
    /// 模型目录布局参考 OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX:
    ///   {ModelPath}/browser_poc_manifest.json
    ///   {ModelPath}/tts_browser_onnx_meta.json
    ///   {ModelPath}/tokenizer.model
    ///   {ModelPath}/moss_tts_prefill.onnx
    ///   {ModelPath}/moss_tts_decode_step.onnx
    ///   {ModelPath}/moss_tts_local_decoder.onnx
    ///   {ModelPath}/moss_tts_local_cached_step.onnx (可选)
    ///   {ModelPath}/moss_tts_local_fixed_sampled_frame.onnx (可选)
    /// 同时需要相邻目录:
    ///   {ModelPath}/../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json
    ///   {ModelPath}/../MOSS-Audio-Tokenizer-Nano-ONNX/moss_audio_tokenizer_*.onnx
    /// </summary>
    public class MossNanoTTSOptions : OnnxExecutionOptions
    {
        /// <summary>
        /// 默认线程数为CPU核心数量一半。但是不能小于2个和大于8个
        /// </summary>
        public override int ThreadsNum { get; set; } = Math.Clamp(Environment.ProcessorCount / 2, 2, 8);

        /// <summary>
        /// MOSS-TTS-Nano-100M-ONNX 模型根目录。
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// 默认音色名称 (manifest builtin_voices[i].voice)。为空时使用 manifest 中第一个音色。
        /// </summary>
        public string? DefaultVoice { get; set; }

        /// <summary>
        /// 单次合成最多生成的 audio frame 数。null 表示沿用 manifest 默认值 (750)。
        /// 模型 frame rate 为 12.5Hz, 即每帧 80ms。750 帧 ≈ 60 秒。
        /// </summary>
        public int? MaxNewFrames { get; set; } = 750;

        /// <summary>
        /// 文本采样温度。null 表示沿用 manifest 默认 (1.0)。
        /// </summary>
        public float? TextTemperature { get; set; } = 1.0f;

        /// <summary>文本 top-k 采样。null 表示沿用 manifest 默认 (50)。</summary>
        public int? TextTopK { get; set; } = 50;

        /// <summary>文本 top-p 采样。null 表示沿用 manifest 默认 (1.0)。</summary>
        public float? TextTopP { get; set; } = 1.0f;

        /// <summary>音频采样温度。null 表示沿用 manifest 默认 (0.7)。</summary>
        public float? AudioTemperature { get; set; } = 0.7f;

        /// <summary>音频 top-k 采样。null 表示沿用 manifest 默认 (25)。</summary>
        public int? AudioTopK { get; set; } = 25;

        /// <summary>音频 top-p 采样。null 表示沿用 manifest 默认 (0.90)。</summary>
        public float? AudioTopP { get; set; } = 0.89f;

        /// <summary>音频 token 重复惩罚系数。null 表示沿用 manifest 默认 (1.2)。</summary>
        public float? AudioRepetitionPenalty { get; set; } = 1.2f;

        /// <summary>
        /// 采样模式。null 表示沿用 manifest 默认 (fixed)。可选 greedy / fixed / full。
        /// fixed = 使用 ONNX 内 fixed_sampled_frame 子图 (默认);
        /// greedy = 贪心解码;
        /// full = 完整 host 端采样 (需要 local_cached_step 子图)。
        ///
        /// 注意: fixed 模式下采样参数 (audio_temperature / top_k / top_p / repetition_penalty) 已在导出时
        /// 固化进 ONNX 子图 (见 tts_meta.fixed_sampled_frame_constants), 本类中对应的 Audio* 配置在 fixed
        /// 模式下不生效; 仅在 full / greedy (host 端采样) 模式下生效。
        /// </summary>
        public MossNanoSampleMode? SampleMode { get; set; } = MossNanoSampleMode.Full;

        /// <summary>
        /// 是否启用采样。null 表示沿用 manifest。false 时强制 greedy。
        /// </summary>
        public bool? DoSample { get; set; } = true;

        /// <summary>
        /// 单段 voice clone 最大文本 token 数 (按 SentencePiece 计)。超过则按句子/逗号切分多段合成。
        /// </summary>
        public int VoiceCloneMaxTextTokens { get; set; } = 75;

        /// <summary>
        /// 默认随机种子。null 表示让 RNG 在调用之间自由演化 (与 Python 默认行为一致, 但每次输出长度会变化)。
        /// 设为非空值时, 每次合成开始前都会重置 RNG, 保证 (text, voice, seed) 决定输出。
        /// 生产环境建议显式设置一个固定种子 (如 1234), 或在调用方按业务需求传 <see cref="Core.Services.TTSService.Models.TtsSynthesisOptions.Seed"/>。
        /// </summary>
        public int? DefaultSeed { get; set; } = 1234;

        /// <summary>
        /// 流式输出时单段累积多少帧再 flush 一次给消费方。范围建议 1~8。默认 4 兼顾首块延迟与吞吐。
        /// 值太小时，若推理速度跟不上音频消费速度，可能出现播放断续。
        /// </summary>
        public int StreamingChunkFrames { get; set; } = 4;

        /// <summary>
        /// 是否使用 WeTextProcessing.NET 对输入文本进行口语化归一。默认启用。
        /// </summary>
        public bool EnableTextNormalization { get; set; } = true;
    }

    /// <summary>
    /// MOSS-TTS-Nano 采样模式。与 ort_cpu_runtime.py 中常量 1:1 对齐。
    /// </summary>
    public enum MossNanoSampleMode
    {
        Greedy = 0,
        Fixed = 1,
        Full = 2,
    }
}
