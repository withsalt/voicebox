using Microsoft.Extensions.Logging;
using System.Diagnostics;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Controllers;

/// <summary>
/// Verifies speaker identity for barge-in and optional idle single-talk candidates. It reports the
/// verdict to the control loop; when identity checks are disabled, candidates pass without inference.
/// </summary>
internal sealed class VadSpeakerGateCoordinator : ISpeakerGateBoundaryPort
{
    private readonly ISpeakerModule? _speaker;
    private readonly ISpeakerVerificationStatePort? _verificationState;
    private readonly ISpeechTimelineLocator _timeline;
    private readonly IVadSpeakerGateHost _host;
    private readonly ILogger _logger;
    private readonly bool _singleTalkEffective;
    private readonly object _deadlineLock = new();
    private long _inferenceReadyTimestamp = -1;
    private long _deadlineVersion = -1;
    private int _lastDeniedStatus = (int)SpeakerVerificationStatus.Unavailable;
    private SpeakerEvidenceQuery _lastQuery;
    private bool _hasLastQuery;
    // -1 when no candidate is active; otherwise the SpeakerVerificationPurpose of the candidate.
    private int _activePurpose = -1;

    public VadSpeakerGateCoordinator(
        ISpeakerModule? speaker,
        IVadSpeakerGateHost host,
        ISpeechTimelineLocator timeline,
        ILogger logger)
    {
        _speaker = speaker;
        _verificationState = speaker;
        _host = host ?? throw new ArgumentNullException(nameof(host));
        _timeline = timeline ?? throw new ArgumentNullException(nameof(timeline));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _singleTalkEffective = speaker is not null &&
            speaker.Policy.Get(SpeakerVerificationPurpose.SingleTalk).Enabled;
    }

    public bool IsIdentityRequired => _speaker is not null;

    public bool IsSingleTalkEffective => _singleTalkEffective;

    public bool HasActiveCandidate => _verificationState?.IsCandidateActive == true;

    public SpeakerVerificationPurpose? ActivePurpose
    {
        get
        {
            int purpose = Volatile.Read(ref _activePurpose);
            return purpose < 0 || !HasActiveCandidate ? null : (SpeakerVerificationPurpose)purpose;
        }
    }

    public SpeakerVerificationStatus LastDeniedStatus
        => (SpeakerVerificationStatus)Volatile.Read(ref _lastDeniedStatus);

    public void BeginCandidate(SpeakerVerificationPurpose purpose)
    {
        Reset();
        if (_verificationState is null)
            return;
        if (purpose == SpeakerVerificationPurpose.SingleTalk && !_singleTalkEffective)
        {
            throw new InvalidOperationException(
                "A single-talk speaker candidate was requested although the SingleTalk purpose is not effective.");
        }

        _verificationState.BeginCandidate(purpose);
        Volatile.Write(ref _activePurpose, (int)purpose);
    }

    public void Reset()
    {
        Volatile.Write(ref _activePurpose, -1);
        _hasLastQuery = false;
        _lastQuery = default;
        _verificationState?.Reset();
        lock (_deadlineLock)
        {
            _inferenceReadyTimestamp = -1;
            _deadlineVersion = -1;
        }
    }

    public BargeInSpeakerGateStatus Evaluate(ReadOnlySpan<float> latestAudio)
    {
        if (_speaker is null || _verificationState is null || !_verificationState.IsCandidateActive)
            return BargeInSpeakerGateStatus.Match;
        long version = _verificationState.CurrentVersion;

        if (!_timeline.TryLocate(
            latestAudio,
            out long streamEpoch,
            out long startSample,
            out long endSample))
        {
            // The identity ring and the processed reservoir share the source timeline; without a
            // locate there is no interval to verify, which is an availability problem, not a mismatch.
            return Deny(
                SpeakerVerificationStatus.Unavailable,
                SpeakerVerificationReason.NotReady,
                latestAudioSamples: latestAudio.Length);
        }

        if (!_verificationState.TryBuildQuery(
                version,
                streamEpoch,
                startSample,
                endSample,
                Stopwatch.GetTimestamp(),
                out SpeakerEvidenceQuery query))
            return Deny(
                SpeakerVerificationStatus.Unavailable,
                SpeakerVerificationReason.CandidateRangeInvalid,
                startSample,
                endSample);

        _lastQuery = query;
        _hasLastQuery = true;

        SpeakerVerificationResult result = _speaker.Evaluate(in query);
        if (!_verificationState.IsCurrent(version))
            return Deny(SpeakerVerificationStatus.Cancelled);

        // Single-talk runs in parallel with ASR while the user is still speaking. Any unavailable
        // result before the endpoint is provisional; resolving it here would deny the utterance
        // mid-sentence (the observed Unavailable/InferenceTimeout with observed=0).
        if (query.Purpose == SpeakerVerificationPurpose.SingleTalk && !query.InputComplete &&
            result.Status == SpeakerVerificationStatus.Unavailable)
        {
            return BargeInSpeakerGateStatus.Pending;
        }

        if (query.Purpose == SpeakerVerificationPurpose.SingleTalk && !query.InputComplete)
            return MapResult(result);

        result = ApplyBoundedDeadline(version, in query, result);
        return MapResult(result);
    }

    public SpeechBoundaryObservation CompleteInput(TimeSpan duration)
    {
        if (_speaker is null)
            return _host.SpeechEndedWithoutSpeakerGate(duration);

        ISpeakerVerificationStatePort state = _verificationState!;
        long version = state.CurrentVersion;
        SpeakerVerificationPurpose purpose = ActivePurpose ?? SpeakerVerificationPurpose.BargeInCommand;
        bool hasAlignedQuery = state.TryCompleteQuery(
            version,
            Stopwatch.GetTimestamp(),
            out SpeakerEvidenceQuery query);

        if (!hasAlignedQuery)
        {
            _hasLastQuery = false;
            _lastQuery = default;
            Resolve(
                version,
                purpose,
                SpeakerVerificationResult.Unavailable(
                    0,
                    in query,
                    SpeakerVerificationReason.CandidateRangeInvalid),
                duration);
            return default;
        }

        _lastQuery = query;
        _hasLastQuery = true;
        SpeakerVerificationResult result = _speaker.Evaluate(in query);
        if (result.IsFinal)
        {
            Resolve(version, purpose, result, duration);
            return default;
        }

        _host.Track(
            "speaker.vad-verification",
            WaitForDecisionAsync(version, purpose, query, duration));
        return default;
    }

    private async Task WaitForDecisionAsync(
        long version,
        SpeakerVerificationPurpose purpose,
        SpeakerEvidenceQuery query,
        TimeSpan duration)
    {
        if (_speaker is null)
            return;

        try
        {
            SpeakerVerificationResult result = await _speaker.WaitForDecisionAsync(
                query,
                _speaker.GetInferenceTimeout(query.Purpose),
                _host.RunToken).ConfigureAwait(false);
            Resolve(version, purpose, result, duration);
        }
        catch (OperationCanceledException) when (_host.IsRunCancelled)
        {
        }
    }

    private void Resolve(
        long version,
        SpeakerVerificationPurpose purpose,
        SpeakerVerificationResult result,
        TimeSpan? pendingSpeechEnd)
    {
        BargeInSpeakerGateStatus status = MapResult(result);
        if (status == BargeInSpeakerGateStatus.Pending)
            return;

        if (!_verificationState!.TryResolve(version))
            return;

        Volatile.Write(ref _activePurpose, -1);
        _host.SubmitGateResolved(_host.NextOperationId(), purpose, status, pendingSpeechEnd);
    }

    /// <summary>
    /// The first complete window is the only thing the realtime Wakeup/BargeIn gate waits for;
    /// once it exists the worker gets the purpose's bounded inference timeout. SingleTalk is
    /// deliberately excluded while its input is open because it verifies in parallel with ASR.
    /// </summary>
    private SpeakerVerificationResult ApplyBoundedDeadline(
        long version,
        in SpeakerEvidenceQuery query,
        SpeakerVerificationResult result)
    {
        if (result.IsFinal || result.Reason != SpeakerVerificationReason.WaitingForInference)
            return result;

        long now = Stopwatch.GetTimestamp();
        lock (_deadlineLock)
        {
            if (_deadlineVersion != version)
            {
                _deadlineVersion = version;
                _inferenceReadyTimestamp = now;
                return result;
            }
            if (_inferenceReadyTimestamp < 0)
            {
                _inferenceReadyTimestamp = now;
                return result;
            }
            if (Stopwatch.GetElapsedTime(_inferenceReadyTimestamp, now) < _speaker!.GetInferenceTimeout(query.Purpose))
                return result;
        }

        return SpeakerVerificationResult.Unavailable(
            result.Threshold,
            in query,
            SpeakerVerificationReason.InferenceTimeout,
            result.RequiredMatches);
    }

    private BargeInSpeakerGateStatus Deny(
        SpeakerVerificationStatus status,
        SpeakerVerificationReason reason = SpeakerVerificationReason.None,
        long startSample = 0,
        long endSample = 0,
        int latestAudioSamples = 0)
    {
        Volatile.Write(ref _lastDeniedStatus, (int)status);
        if (_logger.IsEnabled(LogLevel.Debug))
        {
            _logger.LogDebug(
                "VAD 说话人验证不可用: status={Status}, reason={Reason}, range=[{Start},{End}), latestAudioSamples={LatestAudioSamples}.",
                status,
                reason,
                startSample,
                endSample,
                latestAudioSamples);
        }
        return BargeInSpeakerGateStatus.Rejected;
    }

    private BargeInSpeakerGateStatus MapResult(in SpeakerVerificationResult result)
    {
        switch (result.Status)
        {
            case SpeakerVerificationStatus.Match:
                return BargeInSpeakerGateStatus.Match;

            case SpeakerVerificationStatus.Pending:
                return BargeInSpeakerGateStatus.Pending;

            default:
                _logger.LogDebug(
                "VAD 候选说话人验证拒绝: candidate={CandidateId}, status={Status}, reason={Reason}, score={Score:F4}, threshold={Threshold:F4}, observed={Observed}, matched={Matched}/{Required}, range=[{Start},{End}), inputComplete={InputComplete}.",
                    result.CandidateId,
                    result.Status,
                    result.Reason,
                    result.Score,
                    result.Threshold,
                    result.ObservedWindows,
                    result.MatchedWindows,
                result.RequiredMatches,
                _hasLastQuery ? _lastQuery.CandidateStartSample : 0,
                _hasLastQuery ? _lastQuery.CandidateEndSampleExclusive : 0,
                _hasLastQuery && _lastQuery.InputComplete);
                return Deny(result.Status);
        }
    }
}
