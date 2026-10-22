using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Assistant.Modules.Speaker.State;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Test seam for Confirmation.Enable compositions whose subject is not identity: every candidate
/// is the owner, verified at once. The composition still requires a speaker module; this fake
/// only removes the model and the profile file from the loop. SingleTalk stays off so ordinary
/// listening turns keep the plain utterance path.
/// </summary>
internal sealed class AlwaysOwnerSpeakerModuleFactory : ISpeakerModuleFactory
{
    public AlwaysOwnerSpeakerModule? Last { get; private set; }

    public ISpeakerModule Create(
        SpeakerCaptureDescriptor descriptor,
        int audioSampleRate,
        int audioChannels,
        ILogger logger)
        => Last = new AlwaysOwnerSpeakerModule(descriptor, audioSampleRate);
}

internal sealed class AlwaysOwnerSpeakerModule : ISpeakerModule
{
    private readonly SpeakerVerificationState _state;
    private long _candidates;
    private long _frames;

    public AlwaysOwnerSpeakerModule(SpeakerCaptureDescriptor descriptor, int sampleRate)
    {
        Descriptor = descriptor;
        Policy = SpeakerVerificationPolicy.Create(
            new ScriptedSpeakerService(),
            SpeakerTestProfiles.Options("unused"),
            sampleRate,
            1,
            new SpeakerPurposeSet(true, true, false));
        _state = new SpeakerVerificationState(this);
    }

    public AssistantTimelineRequirement TimelineRequirement => AssistantTimelineRequirement.SpeakerCaptureTimeline;

    public SpeakerCaptureDescriptor Descriptor { get; }

    public SpeakerVerificationPolicy Policy { get; }

    public bool Started { get; private set; }

    public long IdentityFrames => Volatile.Read(ref _frames);

    public List<SpeakerEvidenceQuery> Queries { get; } = [];

    public void Start() => Started = true;

    public void SetPlaybackActive(bool active) { }

    public Task StopAsync(CancellationToken cancellationToken = default)
    {
        Started = false;
        return Task.CompletedTask;
    }

    public void OnSpeakerCapture(in SpeakerCaptureFrame frame) => Interlocked.Increment(ref _frames);

    public SpeakerCandidateId NewCandidateId() => new(Interlocked.Increment(ref _candidates));

    public SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query)
    {
        lock (Queries)
            Queries.Add(query);
        return new SpeakerVerificationResult(
            SpeakerVerificationStatus.Match,
            1f,
            0.5f,
            query.StreamEpoch,
            query.CandidateStartSample,
            query.CandidateEndSampleExclusive,
            SpeakerVerificationReason.HighConfidenceMatch,
            ObservedWindows: 1,
            MatchedWindows: 1,
            RequiredMatches: 1,
            HighConfidenceMatch: true,
            CandidateId: query.CandidateId);
    }

    public Task<SpeakerVerificationResult> WaitForDecisionAsync(
        SpeakerEvidenceQuery query,
        TimeSpan timeout,
        CancellationToken cancellationToken = default)
        => Task.FromResult(Evaluate(in query));

    public void Cancel(SpeakerCandidateId candidateId) { }

    public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose) => TimeSpan.FromMilliseconds(200);

    // The VAD-gate session port: the gate answers Match before it ever needs to locate audio.
    bool ISpeakerVerificationStatePort.IsCandidateActive => false;
    long ISpeakerVerificationStatePort.CurrentVersion => _state.CurrentVersion;
    void ISpeakerVerificationStatePort.BeginCandidate(SpeakerVerificationPurpose purpose) => _state.BeginCandidate(purpose);
    void ISpeakerVerificationStatePort.Reset() => _state.Reset();
    bool ISpeakerVerificationStatePort.TryBuildQuery(long version, long streamEpoch, long startSample, long endSample, long queryTimestampTicks, out SpeakerEvidenceQuery query)
        => _state.TryBuildQuery(version, streamEpoch, startSample, endSample, queryTimestampTicks, out query);
    bool ISpeakerVerificationStatePort.TryCompleteQuery(long version, long queryTimestampTicks, out SpeakerEvidenceQuery query)
        => _state.TryCompleteQuery(version, queryTimestampTicks, out query);
    bool ISpeakerVerificationStatePort.IsCurrent(long version) => _state.IsCurrent(version);
    bool ISpeakerVerificationStatePort.TryResolve(long version) => _state.TryResolve(version);

    public void Dispose() { }
}
