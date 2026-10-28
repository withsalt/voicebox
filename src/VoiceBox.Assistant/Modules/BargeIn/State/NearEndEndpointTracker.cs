using VoiceBox.Assistant.Modules.BargeIn.Buffering;

namespace VoiceBox.Assistant.Modules.BargeIn.State;

/// <summary>
/// Tracks endpoint-VAD audio on the main AEC timeline, including endpoints seen while a DTD
/// candidate is still forming and endpoints recorded by a capture callback the control plane has
/// not acknowledged yet. All mutation happens under the near-end data-plane lock.
/// </summary>
internal sealed class NearEndEndpointTracker
{
    private readonly NearEndTimelineBuffer _timeline;

    public NearEndEndpointTracker(NearEndTimelineBuffer timeline)
        => _timeline = timeline ?? throw new ArgumentNullException(nameof(timeline));

    // Endpoint VAD candidate formed before capture commits.
    public bool CandidateActive { get; private set; }
    public bool CandidateLocated { get; private set; }
    public bool CandidateLocationAttempted { get; private set; }
    public long CandidateGeneration { get; private set; }
    public long CandidateStreamEpoch { get; private set; }
    public long CandidateStartSample { get; private set; }
    public long CandidateEndSample { get; private set; }

    // Endpoint ownership while capture already owns the direct feed.
    public bool CaptureObserved { get; private set; }
    public bool CaptureSpeechActive { get; private set; }

    // Endpoint recorded by a capture callback that the control plane has not acknowledged yet.
    public bool UnownedObserved { get; private set; }
    public bool UnownedProcessed { get; private set; }
    public long UnownedGeneration { get; private set; }
    public long UnownedStreamEpoch { get; private set; }
    public long UnownedEndSample { get; private set; }

    public void BeginCandidate()
    {
        CandidateActive = true;
        CandidateGeneration = _timeline.Generation;
        CandidateStreamEpoch = _timeline.StreamEpoch;
    }

    public void ClearCandidate()
    {
        CandidateActive = false;
        CandidateLocated = false;
        CandidateLocationAttempted = false;
        CandidateGeneration = 0;
        CandidateStreamEpoch = 0;
        CandidateStartSample = 0;
        CandidateEndSample = 0;
    }

    public void RecordUnowned(long generation, long streamEpoch, long endSample)
    {
        UnownedObserved = true;
        UnownedGeneration = generation;
        UnownedStreamEpoch = streamEpoch;
        UnownedEndSample = endSample;
    }

    public void ClearUnowned()
    {
        UnownedObserved = false;
        UnownedProcessed = false;
        UnownedGeneration = 0;
        UnownedStreamEpoch = 0;
        UnownedEndSample = 0;
    }

    public void MarkUnownedProcessed() => UnownedProcessed = true;

    public void ObserveCaptureStarted(bool endpointVadCandidate)
    {
        CaptureObserved = endpointVadCandidate;
        CaptureSpeechActive = endpointVadCandidate;
    }

    public void MarkCaptureSpeechActive()
    {
        CaptureObserved = true;
        CaptureSpeechActive = true;
    }

    public void MarkCaptureSpeechEnded()
    {
        CaptureObserved = true;
        CaptureSpeechActive = false;
    }

    public void ClearCaptureObservation()
    {
        CaptureObserved = false;
        CaptureSpeechActive = false;
    }

    /// <summary>
    /// Extends the located candidate interval with one VAD chunk that was matched against the
    /// main-AEC reservoir.
    /// </summary>
    public void ExtendCandidate(long streamEpoch, long startSample, long endSample)
    {
        CandidateStreamEpoch = streamEpoch;
        if (!CandidateLocated)
        {
            CandidateStartSample = startSample;
            CandidateLocated = true;
        }
        else
        {
            CandidateStartSample = Math.Min(CandidateStartSample, startSample);
        }
        CandidateEndSample = Math.Max(CandidateEndSample, endSample);
    }

    /// <summary>
    /// Locates one VAD chunk on the main-AEC timeline. <see cref="CandidateLocationAttempted"/>
    /// records that a location attempt already happened, which distinguishes "not yet observed"
    /// from "observed but unmatched".
    /// </summary>
    public bool TryLocate(
        ReadOnlySpan<float> audio,
        out long streamEpoch,
        out long startSample,
        out long endSample)
    {
        CandidateLocationAttempted = true;
        return _timeline.TryLocate(audio, out streamEpoch, out startSample, out endSample);
    }
}
