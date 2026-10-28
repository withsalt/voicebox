using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Controllers;

/// <summary>
/// The near-end module's only exit to external collaborators. The data plane calls it strictly
/// after releasing its state lock. Playback commands are sequenced per owner turn: each command
/// is an absolute gain state for one owner, so a stale command is dropped only by a newer command
/// for the same owner. A newer restore for another owner never suppresses the restore of the
/// owner that is actually ducked.
/// </summary>
internal sealed class NearEndEffectExecutor
{
    private const int MaxTrackedPlaybackOwners = 16;

    private readonly IActiveNearEndPlaybackPort _playback;
    private readonly IActiveNearEndSignalSink _control;
    private readonly INearEndRecognitionAudioSink _recognitionAudio;
    private readonly ISpeakerEvidenceProvider? _speakerEvidence;
    private readonly object _playbackGate = new();
    private readonly PlaybackOwnerSequence[] _playbackSequences =
        new PlaybackOwnerSequence[MaxTrackedPlaybackOwners];
    private int _playbackSequenceCount;

    private readonly record struct PlaybackOwnerSequence(long OwnerTurnId, long Sequence);

    public NearEndEffectExecutor(
        IActiveNearEndPlaybackPort playback,
        IActiveNearEndSignalSink control,
        INearEndRecognitionAudioSink recognitionAudio,
        ISpeakerEvidenceProvider? speakerEvidence)
    {
        _playback = playback ?? throw new ArgumentNullException(nameof(playback));
        _control = control ?? throw new ArgumentNullException(nameof(control));
        _recognitionAudio = recognitionAudio ?? throw new ArgumentNullException(nameof(recognitionAudio));
        _speakerEvidence = speakerEvidence;
    }

    public bool IsPlaybackActive => _playback.IsPlaybackActive;

    public long NextOperationId() => _control.NextOperationId();

    public SpeakerCandidateId NewSpeakerCandidateId()
        => _speakerEvidence?.NewCandidateId() ?? SpeakerCandidateId.None;

    public SpeakerVerificationResult EvaluateSpeaker(in SpeakerEvidenceQuery query)
        => _speakerEvidence?.Evaluate(in query)
            ?? SpeakerVerificationResult.Unavailable(
                0,
                in query,
                SpeakerVerificationReason.NotReady);

    public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose)
        => _speakerEvidence?.GetInferenceTimeout(purpose) ?? TimeSpan.Zero;

    public NearEndEffectResult Execute(NearEndEffect effect)
    {
        ArgumentNullException.ThrowIfNull(effect);
        return effect switch
        {
            NearEndEffect.DuckPlayback duck => ExecuteDuck(duck),
            NearEndEffect.RestorePlayback restore => ExecuteRestore(restore),
            NearEndEffect.QueueConfirmation queue => new(_control.TryEnqueueConfirmation(queue.OperationId)),
            NearEndEffect.PublishSpeechEnded ended => PublishSpeechEnded(ended),
            NearEndEffect.CancelSpeakerCandidate cancel => CancelSpeakerCandidate(cancel),
            NearEndEffect.ReportDataPlaneFault fault => ReportDataPlaneFault(fault),
            NearEndEffect.ResetRecognition reset => ResetRecognition(reset),
            _ => NearEndEffectResult.Ok,
        };
    }

    /// <summary>
    /// Synchronous ASR feed. The span never escapes the call, so it cannot be stored in a typed
    /// effect; the executor still remains the single exit to the recognition module.
    /// </summary>
    public NearEndEffectResult FeedRecognition(long turnId, ReadOnlySpan<float> audio)
    {
        bool accepted = _recognitionAudio.TryFeed(turnId, audio);
        return accepted
            ? new NearEndEffectResult(true)
            : new NearEndEffectResult(false, _recognitionAudio.HasAttachedSession(turnId));
    }

    private NearEndEffectResult ExecuteDuck(NearEndEffect.DuckPlayback duck)
    {
        lock (_playbackGate)
        {
            if (!TryAdvancePlaybackSequenceLocked(duck.OwnerTurnId, duck.Sequence))
                return NearEndEffectResult.Ok;
            return new NearEndEffectResult(
                _playback.TryDuckPlayback(duck.OwnerTurnId, duck.Gain, duck.OriginalGain));
        }
    }

    private NearEndEffectResult ExecuteRestore(NearEndEffect.RestorePlayback restore)
    {
        lock (_playbackGate)
        {
            if (TryAdvancePlaybackSequenceLocked(restore.OwnerTurnId, restore.Sequence))
                _playback.TryRestorePlayback(restore.OwnerTurnId);
        }
        return NearEndEffectResult.Ok;
    }

    private NearEndEffectResult PublishSpeechEnded(NearEndEffect.PublishSpeechEnded ended)
    {
        _control.EnqueueSpeechEnded(ended.TurnId, ended.Duration);
        return NearEndEffectResult.Ok;
    }

    private NearEndEffectResult CancelSpeakerCandidate(NearEndEffect.CancelSpeakerCandidate cancel)
    {
        if (!cancel.CandidateId.IsNone)
            _speakerEvidence?.Cancel(cancel.CandidateId);
        return NearEndEffectResult.Ok;
    }

    private NearEndEffectResult ReportDataPlaneFault(NearEndEffect.ReportDataPlaneFault fault)
    {
        _control.ReportDataPlaneFault(fault.Failure);
        return NearEndEffectResult.Ok;
    }

    private NearEndEffectResult ResetRecognition(NearEndEffect.ResetRecognition reset)
    {
        _recognitionAudio.Reset(reset.TurnId);
        return NearEndEffectResult.Ok;
    }

    /// <summary>
    /// Applies playback intents in per-owner issuance order while the gate is held. Each command
    /// is an absolute gain state for one owner, so an older command is dropped only when a newer
    /// command for the same owner was already applied. Restores for different owners remain
    /// independent; the gate also keeps the sequence check and the playback call atomic.
    /// </summary>
    private bool TryAdvancePlaybackSequenceLocked(long ownerTurnId, long sequence)
    {
        for (int index = 0; index < _playbackSequenceCount; index++)
        {
            if (_playbackSequences[index].OwnerTurnId != ownerTurnId)
                continue;

            if (sequence <= _playbackSequences[index].Sequence)
                return false;

            _playbackSequences[index] = new PlaybackOwnerSequence(ownerTurnId, sequence);
            return true;
        }

        if (_playbackSequenceCount < _playbackSequences.Length)
        {
            _playbackSequences[_playbackSequenceCount++] =
                new PlaybackOwnerSequence(ownerTurnId, sequence);
            return true;
        }

        // Bounded owner tracking: evict the owner with the oldest command. Evicted owners are
        // long-idle playback requests whose stale commands cannot affect the current reply.
        int oldest = 0;
        for (int index = 1; index < _playbackSequences.Length; index++)
        {
            if (_playbackSequences[index].Sequence < _playbackSequences[oldest].Sequence)
                oldest = index;
        }
        _playbackSequences[oldest] = new PlaybackOwnerSequence(ownerTurnId, sequence);
        return true;
    }
}
