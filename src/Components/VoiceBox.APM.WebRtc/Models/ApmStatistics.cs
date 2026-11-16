namespace VoiceBox.APM.WebRtc.Models
{
    /// <summary>
    /// APM 统计快照(对外只读视图)。各指标可能在收敛前缺失，<c>Has*</c> 为 false 时对应值无意义。
    /// 对应原生 <c>ApmStats</c> / WebRTC <c>AudioProcessingStats</c>。
    /// </summary>
    public readonly record struct ApmStatistics(
        bool HasEchoReturnLoss,
        double EchoReturnLoss,
        bool HasEchoReturnLossEnhancement,
        double EchoReturnLossEnhancement,
        bool HasDelayMs,
        int DelayMs,
        bool HasDelayMedianMs,
        int DelayMedianMs,
        bool HasResidualEchoLikelihood,
        double ResidualEchoLikelihood,
        bool HasResidualEchoLikelihoodRecentMax = false,
        double ResidualEchoLikelihoodRecentMax = 0,
        bool HasDivergentFilterFraction = false,
        double DivergentFilterFraction = 0);
}
