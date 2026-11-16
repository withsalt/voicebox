namespace VoiceBox.Core.Services.SpeakerService.Models
{
    /// <summary>流式说话人验证的单窗口结果。</summary>
    public class SpeakerEventArgs : EventArgs
    {
        /// <summary>窗口结束时刻 (自流开始/上次 Reset 起的音频时间)。</summary>
        public TimeSpan AudioTime { get; }

        /// <summary>得分最高的已注册说话人; 无注册说话人时为 null。</summary>
        public string? BestSpeaker { get; }

        /// <summary>最高说话人的平滑得分; 无注册说话人时为 0。</summary>
        public float BestScore { get; }

        /// <summary>最高得分是否超过判定阈值。</summary>
        public bool IsMatch { get; }

        /// <summary>各已注册说话人的平滑得分。</summary>
        public IReadOnlyDictionary<string, float> Scores { get; }

        /// <summary>本窗口的说话人 embedding (L2 归一化), 可用于自定义比对/聚类。</summary>
        public float[] Embedding { get; }

        /// <summary>
        /// 是否为前缀窗提前打分 (early-exit) 结果: 首个完整窗就绪前, 对不足整窗的前缀音频
        /// [流起点, <see cref="AudioTime"/>) 推理产出。得分为原始值 (不参与平滑), 可靠性低于
        /// 完整窗结果, 消费方应采用更严格的接受策略且不应据此提前否定。
        /// </summary>
        public bool IsEarlyExit { get; }

        public SpeakerEventArgs(
            TimeSpan audioTime,
            string? bestSpeaker,
            float bestScore,
            bool isMatch,
            IReadOnlyDictionary<string, float> scores,
            float[] embedding,
            bool isEarlyExit = false)
        {
            AudioTime = audioTime;
            BestSpeaker = bestSpeaker;
            BestScore = bestScore;
            IsMatch = isMatch;
            Scores = scores;
            Embedding = embedding;
            IsEarlyExit = isEarlyExit;
        }
    }
}
