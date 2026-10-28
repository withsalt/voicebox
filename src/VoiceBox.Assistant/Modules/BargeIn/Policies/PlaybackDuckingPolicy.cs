using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.Policies;

/// <summary>
/// Computes owner-scoped playback gain intents for the acoustic-only path. It never touches the
/// playback device: the data plane executes the returned commands outside its state lock, and the
/// monotonic sequence makes a stale duck/restore a no-op once a newer intent was issued.
/// All mutation happens under the near-end data-plane lock.
/// </summary>
internal sealed class PlaybackDuckingPolicy
{
    public enum Kind
    {
        None,
        Duck,
        Restore,
    }

    public readonly record struct Command(
        Kind Kind,
        long Sequence,
        long OwnerTurnId,
        long StreamEpoch,
        float Gain,
        float OriginalGain,
        long LastSpeechEndSample)
    {
        public static Command None => default;
    }

    /// <summary>At most one restore followed by one duck can be produced by a single frame.</summary>
    public readonly record struct CommandBatch(Command First, Command Second)
    {
        public static CommandBatch None => default;

        public bool IsEmpty => First.Kind == Kind.None && Second.Kind == Kind.None;

        public bool ContainsDuck => First.Kind == Kind.Duck || Second.Kind == Kind.Duck;
    }

    private long _duckedOwnerTurnId;
    private long _duckedStreamEpoch;
    private long _lastSpeechEndSample;
    private long _sequence;

    public long Sequence => _sequence;

    public bool IsDucked => _duckedOwnerTurnId != 0;

    public CommandBatch Update(
        in DuplexAudioFrame frame,
        in BargeInDecision decision,
        bool candidateActive,
        bool confirmationPending,
        bool allowsDucking,
        long ownerTurnId,
        float duckGain,
        int releaseMs)
    {
        Command first = Command.None;
        if (_duckedOwnerTurnId != 0 &&
            (_duckedOwnerTurnId != ownerTurnId || _duckedStreamEpoch != frame.StreamEpoch))
        {
            first = Restore();
        }

        if (!allowsDucking ||
            frame.HasDiscontinuity ||
            !NearEndDecisionRules.HasUsableTimeline(in decision) ||
            decision.StreamEpoch != frame.StreamEpoch ||
            duckGain >= 1)
        {
            return new CommandBatch(first, Restore());
        }

        // Once DTD confirms, keep the reduced gain until the owner stops and clears its audio.
        // Do not briefly restore the old reply while the control loop catches up.
        if (candidateActive || confirmationPending || decision.Action == BargeInAction.Interrupt)
            return new CommandBatch(first, Command.None);

        long releaseSamples = AssistantAudioFormat.SamplesFromMilliseconds(
            releaseMs,
            checked((int)frame.SampleRate),
            checked((int)frame.Channels));
        bool recentSpeech = NearEndDecisionRules.HasNearEndSpeechEvidence(in decision) &&
            frame.EndSampleExclusive - decision.EndSampleExclusive <= releaseSamples;
        if (recentSpeech)
        {
            if (_duckedOwnerTurnId == 0)
            {
                long sequence = checked(++_sequence);
                _duckedOwnerTurnId = ownerTurnId;
                _duckedStreamEpoch = frame.StreamEpoch;
                _lastSpeechEndSample = decision.EndSampleExclusive;
                return new CommandBatch(
                    first,
                    new Command(
                        Kind.Duck,
                        sequence,
                        ownerTurnId,
                        frame.StreamEpoch,
                        duckGain,
                        frame.PlaybackGain,
                        decision.EndSampleExclusive));
            }

            _lastSpeechEndSample = Math.Max(_lastSpeechEndSample, decision.EndSampleExclusive);
            return new CommandBatch(first, Command.None);
        }

        if (_duckedOwnerTurnId != 0 &&
            frame.EndSampleExclusive - _lastSpeechEndSample >= releaseSamples)
        {
            return new CommandBatch(first, Restore());
        }

        return new CommandBatch(first, Command.None);
    }

    /// <summary>Clears the ducked state and returns the restore intent, if one is outstanding.</summary>
    public Command Restore()
    {
        if (_duckedOwnerTurnId == 0)
            return Command.None;

        long sequence = checked(++_sequence);
        var command = new Command(
            Kind.Restore,
            sequence,
            _duckedOwnerTurnId,
            _duckedStreamEpoch,
            0,
            0,
            0);
        _duckedOwnerTurnId = 0;
        _duckedStreamEpoch = 0;
        _lastSpeechEndSample = 0;
        return command;
    }

    /// <summary>
    /// Clears the optimistic duck state when the playback port rejected the request and no newer
    /// command has been issued since.
    /// </summary>
    public void RollbackDuck(long sequence)
    {
        if (_sequence != sequence)
            return;

        _duckedOwnerTurnId = 0;
        _duckedStreamEpoch = 0;
        _lastSpeechEndSample = 0;
    }
}
