using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Kernel.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers
{
    /// <summary>
    /// Keeps per-turn barge-in metadata until ASR validation completes.
    /// ASR outcomes are deliberately not fed back into the acoustic decision model.
    /// </summary>
    internal sealed class BargeInDecisionJournal
    {
        private readonly object _triggerSourceLock = new();
        private readonly Dictionary<long, BargeInTriggerSource> _triggerSources = new();
        private readonly object _decisionLock = new();
        private readonly Dictionary<long, BargeInDecisionDiagnostics> _decisionContexts = new();

        public void RememberTriggerSource(long turnId, BargeInTriggerSource source)
        {
            if (turnId == 0)
                return;

            lock (_triggerSourceLock)
                _triggerSources[turnId] = source;
        }

        public BargeInTriggerSource GetTriggerSource(long turnId)
        {
            lock (_triggerSourceLock)
                return _triggerSources.TryGetValue(turnId, out BargeInTriggerSource source)
                    ? source
                    : BargeInTriggerSource.Vad;
        }

        public void ForgetTriggerSource(long turnId)
        {
            lock (_triggerSourceLock)
                _triggerSources.Remove(turnId);
        }

        public void RememberContext(
            long turnId,
            BargeInDecisionDiagnostics diagnostics)
        {
            if (turnId == 0)
                return;

            lock (_decisionLock)
                _decisionContexts[turnId] = diagnostics;
        }

        public bool TryGetDiagnostics(long turnId, out BargeInDecisionDiagnostics diagnostics)
        {
            lock (_decisionLock)
            {
                if (_decisionContexts.TryGetValue(turnId, out diagnostics))
                {
                    return true;
                }
            }

            diagnostics = default;
            return false;
        }

        public bool CompleteContext(long turnId)
            => TryTakeContext(turnId);

        public void Clear()
        {
            lock (_triggerSourceLock)
                _triggerSources.Clear();
            lock (_decisionLock)
                _decisionContexts.Clear();
        }

        private bool TryTakeContext(long turnId)
        {
            lock (_decisionLock)
                return _decisionContexts.Remove(turnId);
        }
    }
}
