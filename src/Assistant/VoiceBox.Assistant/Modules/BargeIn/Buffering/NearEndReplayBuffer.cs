using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Runtime.Control.Queues;

namespace VoiceBox.Assistant.Modules.BargeIn.Buffering;

/// <summary>
/// Bounded confirmation-tail metadata plus the single-consumer drain lease. The queue never holds
/// PCM: frames reference the timeline reservoir, which is replayed exactly once after the control
/// plane commits the capture. Guarded by the near-end data-plane lock; the drain lease is the only
/// lock-free gate.
/// </summary>
internal sealed class NearEndReplayBuffer
{
    private const int Capacity = 256;

    private readonly BoundedMpscQueue<NearEndConfirmationTailFrame> _frames = new(Capacity);
    private int _drainActive;
    private long _nextSample;

    public bool DrainActive => Volatile.Read(ref _drainActive) != 0;

    /// <summary>First sample not yet captured into a tail frame.</summary>
    public long NextSample => _nextSample;

    public bool TryWrite(in NearEndConfirmationTailFrame frame) => _frames.TryWrite(frame);

    public bool TryPeek(out NearEndConfirmationTailFrame frame) => _frames.TryPeek(out frame);

    public bool HasFrames => _frames.TryPeek(out _);

    public bool TryRead(out NearEndConfirmationTailFrame frame) => _frames.TryRead(out frame);

    public void AdvanceNextSample(long endSampleExclusive)
        => _nextSample = Math.Max(_nextSample, endSampleExclusive);

    /// <summary>Resets the replay watermark to an immutable confirmation boundary.</summary>
    public void ResetNextSample(long startSample) => _nextSample = startSample;

    public bool TryEnterDrain()
        => Interlocked.CompareExchange(ref _drainActive, 1, 0) == 0;

    public void ExitDrain() => Volatile.Write(ref _drainActive, 0);

    public void Clear()
    {
        while (_frames.TryRead(out _))
        {
        }
        _nextSample = 0;
    }
}
