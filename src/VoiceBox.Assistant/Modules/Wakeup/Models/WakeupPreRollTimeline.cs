using VoiceBox.Assistant.Modules.Wakeup.Buffering;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Assistant.Modules.Wakeup.Models;

/// <summary>
/// Bounded wake pre-roll plus the capture-timeline anchors used to align KWS estimates and to
/// slice replay audio. Mutated only under the gate's state lock.
/// </summary>
internal sealed class WakeupPreRollTimeline
{
    private readonly WakeupAudioBuffer _preRoll;
    private readonly int _sampleRate;
    private readonly int _channels;
    private readonly int _replayLeadInMs;

    private long _wakeDetectorStartSampleIndex;
    private long _preRollStartSampleIndex;

    public WakeupPreRollTimeline(int preRollSamples, int sampleRate, int channels, int replayLeadInMs)
    {
        _preRoll = new WakeupAudioBuffer(preRollSamples);
        _sampleRate = sampleRate;
        _channels = channels;
        _replayLeadInMs = replayLeadInMs;
    }

    public long TotalSamplesSeen { get; private set; }

    public long WakeDetectorStartCaptureSample { get; private set; }

    public long WakeDetectorStreamEpoch { get; private set; }

    public long PreRollStartCaptureSample { get; private set; }

    public bool OriginPending { get; private set; } = true;

    public void AdvanceSamplesSeen(int sampleCount) => TotalSamplesSeen += sampleCount;

    public void SetOrigin(in CaptureRange range)
    {
        _wakeDetectorStartSampleIndex = TotalSamplesSeen;
        WakeDetectorStartCaptureSample = range.StartSample;
        WakeDetectorStreamEpoch = range.StreamEpoch;
        PreRollStartCaptureSample = range.StartSample;
        OriginPending = false;
    }

    public void ClearForRestart()
    {
        _preRoll.Clear();
        _wakeDetectorStartSampleIndex = TotalSamplesSeen;
        _preRollStartSampleIndex = TotalSamplesSeen;
        OriginPending = true;
    }

    public void ClearForDiscontinuity(in CaptureRange range)
    {
        _preRoll.Clear();
        _preRollStartSampleIndex = TotalSamplesSeen;
        SetOrigin(in range);
    }

    public void ClearForStart()
    {
        _preRoll.Clear();
        _wakeDetectorStartSampleIndex = TotalSamplesSeen;
        _preRollStartSampleIndex = TotalSamplesSeen;
        OriginPending = true;
    }

    public void ClearForStop()
    {
        _preRoll.Clear();
        _preRollStartSampleIndex = TotalSamplesSeen;
        OriginPending = true;
    }

    public void Append(ReadOnlySpan<float> audio, long captureEndSample)
    {
        _preRoll.Append(audio);
        _preRollStartSampleIndex = TotalSamplesSeen - _preRoll.Count;
        PreRollStartCaptureSample = captureEndSample - _preRoll.Count;
    }

    /// <summary>Re-anchors both timelines to a consumed or rejected wake event.</summary>
    public void AnchorAt(long captureEndSample)
    {
        _wakeDetectorStartSampleIndex = TotalSamplesSeen;
        WakeDetectorStartCaptureSample = captureEndSample;
        _preRollStartSampleIndex = TotalSamplesSeen;
        PreRollStartCaptureSample = captureEndSample;
    }

    public void Clear() => _preRoll.Clear();

    public float[] TakeReplayAudio(
        ReadOnlySpan<float> trailingAudio,
        WakeupEventArgs wakeup,
        WakeupTriggerKind triggerKind)
        => SliceReplayAudio(
            AppendTrailingAudio(_preRoll.SnapshotAndClear(), trailingAudio),
            wakeup,
            triggerKind);

    public long ComputeWakeStartCaptureSample(WakeupEventArgs wakeup)
        => checked(
            WakeDetectorStartCaptureSample +
            wakeup.StartTimeMs * _sampleRate * _channels / 1000);

    public long ComputeWakeEndCaptureSample(long wakeStartSample, WakeupEventArgs wakeup)
    {
        long durationMs = Math.Max(0L, wakeup.DurationMs);
        return checked(wakeStartSample + durationMs * _sampleRate * _channels / 1000);
    }

    /// <summary>
    /// Keep a small amount of audio before the KWS estimate available for window alignment, but
    /// tell the tracker the detector's exact interval so it does not anchor evidence to the
    /// growing trailing pre-roll.
    /// </summary>
    public long ComputeCandidateStartCaptureSample(long wakeStartSample)
        => Math.Max(
            PreRollStartCaptureSample,
            wakeStartSample - (long)_replayLeadInMs * _sampleRate * _channels / 1000);

    private float[] SliceReplayAudio(float[] audio, WakeupEventArgs wakeup, WakeupTriggerKind triggerKind)
    {
        if (audio.Length == 0)
            return audio;

        long replayStartMs = triggerKind == WakeupTriggerKind.InSessionWakeup
            ? wakeup.StartTimeMs
            : wakeup.StartTimeMs - _replayLeadInMs;
        long wakeStartSample = _wakeDetectorStartSampleIndex + replayStartMs * _sampleRate * _channels / 1000;
        long offset = wakeStartSample - _preRollStartSampleIndex;
        if (offset <= 0)
            return audio;
        if (offset >= audio.Length)
            return Array.Empty<float>();
        return audio.AsSpan((int)offset).ToArray();
    }

    private static float[] AppendTrailingAudio(float[] buffered, ReadOnlySpan<float> trailing)
    {
        if (trailing.IsEmpty)
            return buffered;
        if (buffered.Length == 0)
            return trailing.ToArray();
        var combined = new float[buffered.Length + trailing.Length];
        buffered.CopyTo(combined, 0);
        trailing.CopyTo(combined.AsSpan(buffered.Length));
        return combined;
    }
}
