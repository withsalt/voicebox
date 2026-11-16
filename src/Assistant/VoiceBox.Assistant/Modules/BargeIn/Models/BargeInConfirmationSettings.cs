namespace VoiceBox.Assistant.Modules.BargeIn.Models
{
    /// <summary>
    /// 单次 barge-in 候选的确认参数。
    /// </summary>
    internal readonly record struct BargeInConfirmationSettings
    {
        public BargeInConfirmationSettings(
            int minDurationMs,
            float minRms,
            int initialSuppressionMs = 0,
            int recognitionSuppressionMs = 0,
            int maxCandidateMs = 0,
            int recognitionAudioLimitMs = 0)
        {
            if (minDurationMs < 0)
                throw new ArgumentOutOfRangeException(nameof(minDurationMs), "Minimum duration must be >= 0.");
            if (!float.IsFinite(minRms) || minRms < 0)
                throw new ArgumentOutOfRangeException(nameof(minRms), "Minimum RMS must be finite and >= 0.");
            if (initialSuppressionMs < 0)
                throw new ArgumentOutOfRangeException(nameof(initialSuppressionMs), "Initial suppression must be >= 0.");
            if (recognitionSuppressionMs < 0)
                throw new ArgumentOutOfRangeException(nameof(recognitionSuppressionMs), "Recognition suppression must be >= 0.");
            if (maxCandidateMs < 0)
                throw new ArgumentOutOfRangeException(nameof(maxCandidateMs), "Maximum candidate duration must be >= 0.");
            if (recognitionAudioLimitMs < 0)
                throw new ArgumentOutOfRangeException(nameof(recognitionAudioLimitMs), "Recognition audio limit must be >= 0.");

            MinDurationMs = minDurationMs;
            MinRms = minRms;
            InitialSuppressionMs = initialSuppressionMs;
            RecognitionSuppressionMs = recognitionSuppressionMs;
            MaxCandidateMs = maxCandidateMs;
            RecognitionAudioLimitMs = recognitionAudioLimitMs;
        }

        public int MinDurationMs { get; }
        public float MinRms { get; }
        /// <summary>候选开头多少毫秒不参与确认门槛计算。</summary>
        public int InitialSuppressionMs { get; }
        /// <summary>候选开头多少毫秒不补喂 ASR。</summary>
        public int RecognitionSuppressionMs { get; }
        /// <summary>候选最长累计多少毫秒仍未确认就原地重开。0 表示不限制。</summary>
        public int MaxCandidateMs { get; }
        /// <summary>候选确认后最多补喂 ASR 的最近音频毫秒数。0 表示不限制。</summary>
        public int RecognitionAudioLimitMs { get; }
    }
}
