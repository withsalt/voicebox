using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.Speaker.State;

/// <summary>
/// SpeakerModule-owned mutable VAD-gate verification session. Versioning makes an in-flight
/// result harmless after a candidate reset or a new capture timeline; each session carries its own
/// candidate id so late inference jobs can be cancelled precisely.
/// </summary>
internal sealed class SpeakerVerificationState : ISpeakerVerificationStatePort
{
    private readonly ISpeakerEvidenceProvider _evidence;
    private readonly object _lock = new();
    private SpeakerEvidenceQuery? _query;
    private SpeakerVerificationPurpose _purpose;
    private SpeakerCandidateId _candidateId;
    private long _version;
    private bool _candidateActive;

    /// <param name="evidence">Issues candidate ids and cancels a superseded candidate's inference.</param>
    public SpeakerVerificationState(ISpeakerEvidenceProvider evidence)
        => _evidence = evidence ?? throw new ArgumentNullException(nameof(evidence));

    public bool IsCandidateActive
    {
        get { lock (_lock) return _candidateActive; }
    }

    public long CurrentVersion
    {
        get { lock (_lock) return _version; }
    }

    public void BeginCandidate(SpeakerVerificationPurpose purpose)
    {
        SpeakerCandidateId previous;
        lock (_lock)
        {
            previous = _candidateId;
            _version = checked(_version + 1);
            _candidateActive = true;
            _purpose = purpose;
            _candidateId = _evidence.NewCandidateId();
            _query = null;
        }
        if (!previous.IsNone)
            _evidence.Cancel(previous);
    }

    public void Reset()
    {
        SpeakerCandidateId previous;
        lock (_lock)
        {
            previous = _candidateId;
            _version = checked(_version + 1);
            _candidateActive = false;
            _candidateId = default;
            _query = null;
        }
        if (!previous.IsNone)
            _evidence.Cancel(previous);
    }

    public bool TryBuildQuery(
        long version,
        long streamEpoch,
        long startSample,
        long endSample,
        long queryTimestampTicks,
        out SpeakerEvidenceQuery query)
    {
        lock (_lock)
        {
            query = default;
            if (!_candidateActive || version != _version)
                return false;

            if (_query is { } existing)
            {
                if (existing.StreamEpoch != streamEpoch || startSample > existing.CandidateEndSampleExclusive)
                    return false;
                startSample = existing.CandidateStartSample;
                endSample = Math.Max(endSample, existing.CandidateEndSampleExclusive);
            }

            query = new SpeakerEvidenceQuery(
                _purpose,
                streamEpoch,
                startSample,
                endSample,
                queryTimestampTicks,
                CandidateId: _candidateId);
            _query = query;
            return true;
        }
    }

    public bool TryCompleteQuery(long version, long queryTimestampTicks, out SpeakerEvidenceQuery query)
    {
        lock (_lock)
        {
            query = default;
            if (!_candidateActive || version != _version || _query is not { } current)
                return false;

            query = current with
            {
                InputComplete = true,
                QueryTimestampTicks = queryTimestampTicks,
            };
            _query = query;
            return true;
        }
    }

    public bool IsCurrent(long version)
    {
        lock (_lock)
            return _candidateActive && version == _version;
    }

    public bool TryResolve(long version)
    {
        SpeakerCandidateId previous;
        lock (_lock)
        {
            if (!_candidateActive || version != _version)
                return false;

            previous = _candidateId;
            _version = checked(_version + 1);
            _candidateActive = false;
            _candidateId = default;
            _query = null;
        }
        if (!previous.IsNone)
            _evidence.Cancel(previous);
        return true;
    }
}
