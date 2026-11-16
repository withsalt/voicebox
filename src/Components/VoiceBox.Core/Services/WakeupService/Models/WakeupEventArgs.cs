namespace VoiceBox.Core.Services.WakeupService.Models
{
    public class WakeupEventArgs : EventArgs
    {
        public string Keyword { get; }

        public int Score { get; }

        public int Threshold { get; }

        public long DurationMs { get; }

        public long StartTimeMs { get; }

        public WakeupEventArgs(string keyword, int score, int threshold, long durationMs, long startTimeMs)
        {
            Keyword = keyword;
            Score = score;
            Threshold = threshold;
            DurationMs = durationMs;
            StartTimeMs = startTimeMs;
        }
    }
}
