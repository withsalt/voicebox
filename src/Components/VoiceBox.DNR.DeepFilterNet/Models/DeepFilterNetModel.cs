namespace VoiceBox.DNR.DeepFilterNet.Models
{
    /// <summary>
    /// DeepFilterNet 模型档位。各档位为一个目录，内含 <c>combined_streaming.onnx</c>（带外置 GRU 状态的
    /// 有状态流式模型）与 <c>config.ini</c>。DeepFilterNet 原生工作于 48 kHz、hop=480（10 ms）。
    /// </summary>
    public enum DeepFilterNetModel
    {
        /// <summary>
        /// DeepFilterNet3 低延迟版（df_lookahead=0，算法延迟约 10 ms）。
        /// emb/df 隐藏维 512、df 层数 3，质量与延迟兼顾，适合实时降噪。
        /// </summary>
        Dfn3LowLatency = 0,
    }

    internal static class DeepFilterNetModelExtensions
    {
        /// <summary>返回模型对应的目录名（位于 runtimes/models/dnr/deepfilternet/ 下）。</summary>
        public static string ToDirName(this DeepFilterNetModel model) => model switch
        {
            DeepFilterNetModel.Dfn3LowLatency => "dfn3_ll",
            _ => throw new ArgumentOutOfRangeException(nameof(model), model, "Unknown DeepFilterNet model."),
        };
    }
}
