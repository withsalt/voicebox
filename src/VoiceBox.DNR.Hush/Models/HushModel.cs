namespace VoiceBox.DNR.Hush.Models
{
    /// <summary>
    /// Hush 模型档位。各档位为一个目录，内含 <c>combined_streaming.onnx</c>（带外置 GRU 状态的
    /// 有状态流式模型）与 <c>config.ini</c>。Hush 原生工作于 16 kHz、hop=160（10 ms）。
    /// </summary>
    public enum HushModel
    {
        /// <summary>
        /// Hush（DeepFilterNet3-SE 的 16 kHz 适配版，df_lookahead=0，算法延迟约 10 ms）。
        /// emb/df 隐藏维 256、df 层数 3、参数约 1.8M，专为语音 Agent / 电话语音的实时降噪
        /// （含背景人声抑制）设计。
        /// </summary>
        Hush16k = 0,
    }
}
