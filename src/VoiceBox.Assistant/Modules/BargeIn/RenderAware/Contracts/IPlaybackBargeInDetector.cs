using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;

internal interface IPlaybackBargeInDetector : IActiveNearEndDecisionSource, IDisposable
{
    DtdProcessBatch Push(
        in DuplexAudioFrame frame,
        RenderDelayHint? primaryAecDelay,
        bool decisionEnabled,
        bool echoLearningEnabled);

    RenderAwareBargeInSnapshot GetSnapshot();

    void ResetDecisionState();

    void ResetStream();
}
