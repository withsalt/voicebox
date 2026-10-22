namespace VoiceBox.Core.Services.TTSService.Models
{
    public sealed class TtsSynthesisOptions
    {
        /// <summary>
        /// 内置音色名称。为空时使用默认音色。
        /// </summary>
        public string? Voice { get; set; }

        /// <summary>
        /// 语速倍率。null 表示沿用 TTS 服务默认值；1.0 表示正常语速。
        /// 仅在具体 TTS 实现支持时生效。
        /// </summary>
        public float? Speed { get; set; }

        /// <summary>
        /// 是否启用采样。null 表示沿用 manifest 默认值，false 表示贪心解码 (greedy)。
        /// </summary>
        public bool? DoSample { get; set; }

        /// <summary>
        /// 单次合成最多生成多少音频帧。null 表示沿用 manifest 的默认值。
        /// </summary>
        public int? MaxNewFrames { get; set; }

        /// <summary>
        /// 随机种子。null 表示沿用当前 RNG 状态。
        /// </summary>
        public int? Seed { get; set; }

        /// <summary>
        /// 语言代码。例如 "auto" / "english" / "chinese" / "japanese" 等。
        /// 为空时由具体实现决定默认值。Qwen3-TTS 使用，其他忽略。
        /// </summary>
        public string? Language { get; set; }

        /// <summary>
        /// 风格指令文本 (例如: "用平和的语气朗读")。仅在模型支持时生效，
        /// 否则将被忽略。Qwen3-TTS 1.7B 模型支持，MossNano 不支持。
        /// </summary>
        public string? Instruct { get; set; }

        /// <summary>
        /// 文本采样温度，覆盖 manifest 默认。
        /// </summary>
        public float? TextTemperature { get; set; }

        public int? TextTopK { get; set; }

        public float? TextTopP { get; set; }

        /// <summary>
        /// 音频采样温度，覆盖 manifest 默认。
        /// </summary>
        public float? AudioTemperature { get; set; }

        public int? AudioTopK { get; set; }

        public float? AudioTopP { get; set; }

        public float? AudioRepetitionPenalty { get; set; }
    }
}
