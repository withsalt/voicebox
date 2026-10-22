using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Runtime.Contracts;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies
{
    /// <summary>
    /// 根据助手所处阶段选择 barge-in 确认策略。播放阶段更保守, 用于过滤 TTS 回放残响。
    /// </summary>
    internal sealed class BargeInConfirmationPolicy
    {
        private readonly BargeInConfirmationSettings _thinking;
        private readonly BargeInConfirmationSettings _speaking;
        private readonly IPlaybackStartupGuardView? _startupGuard;

        public BargeInConfirmationPolicy(
            BargeInConfirmationSettings thinking,
            BargeInConfirmationSettings speaking,
            IPlaybackStartupGuardView? startupGuard = null)
        {
            _thinking = thinking;
            _speaking = speaking;
            _startupGuard = startupGuard;
        }

        public bool TryGetSettings(AssistantState state, out BargeInConfirmationSettings settings)
        {
            switch (state)
            {
                case AssistantState.Thinking:
                    settings = _thinking;
                    return true;
                case AssistantState.Speaking:
                    settings = CreateSpeakingSettings();
                    return true;
                default:
                    settings = default;
                    return false;
            }
        }

        private BargeInConfirmationSettings CreateSpeakingSettings()
        {
            int startupSuppressionMs = _startupGuard?.RemainingMilliseconds ?? 0;
            int initialSuppressionMs = Math.Max(_speaking.InitialSuppressionMs, startupSuppressionMs);
            int recognitionSuppressionMs = Math.Max(_speaking.RecognitionSuppressionMs, startupSuppressionMs);

            if (initialSuppressionMs == _speaking.InitialSuppressionMs &&
                recognitionSuppressionMs == _speaking.RecognitionSuppressionMs)
                return _speaking;

            return new BargeInConfirmationSettings(
                _speaking.MinDurationMs,
                _speaking.MinRms,
                initialSuppressionMs,
                recognitionSuppressionMs,
                _speaking.MaxCandidateMs,
                _speaking.RecognitionAudioLimitMs);
        }
    }
}
