using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Exceptions;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.Speaker.Enrollment;

/// <summary>
/// Collects one enrollment segment from the identity tap. Speech audio is what the VAD reports as
/// speech; total samples, render activity and DTD near-end acceptance are counted per identity
/// frame so the quality evaluator can compute ratios. A timeline gap fails the attempt at once.
/// </summary>
internal sealed class SpeakerEnrollmentSegmentCollector
{
    private readonly object _lock = new();
    private readonly List<float> _speech = [];
    private TaskCompletionSource<SpeakerEnrollmentSegmentCapture>? _completion;
    private CaptureRange? _previousRange;
    private long _totalSamples;
    private int _frameCount;
    private int _renderActiveFrames;
    private int _nearEndAcceptedFrames;
    private bool _nearEndObserved;
    private int _requiredSpeechSamples;
    private int _lastReportedPercent = -1;

    public event Action<int, int>? Progress;

    public bool IsActive
    {
        get { lock (_lock) return _completion is not null; }
    }

    public Task<SpeakerEnrollmentSegmentCapture> Begin(int requiredSpeechSamples)
    {
        if (requiredSpeechSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(requiredSpeechSamples));
        var completion = new TaskCompletionSource<SpeakerEnrollmentSegmentCapture>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        lock (_lock)
        {
            ResetLocked();
            _requiredSpeechSamples = requiredSpeechSamples;
            _completion = completion;
        }
        Progress?.Invoke(0, requiredSpeechSamples);
        return completion.Task;
    }

    /// <summary>Ends the attempt; a still-incomplete segment is reported with what it has.</summary>
    public SpeakerEnrollmentSegmentCapture? End()
    {
        TaskCompletionSource<SpeakerEnrollmentSegmentCapture>? completion;
        SpeakerEnrollmentSegmentCapture? partial;
        lock (_lock)
        {
            completion = _completion;
            partial = completion is null ? null : SnapshotLocked();
            ResetLocked();
        }
        if (completion is not null && partial is not null)
            completion.TrySetResult(partial);
        return partial;
    }

    public void Fail(Exception exception)
    {
        TaskCompletionSource<SpeakerEnrollmentSegmentCapture>? completion;
        lock (_lock)
        {
            completion = _completion;
            _completion = null;
        }
        completion?.TrySetException(exception);
    }

    /// <summary>Identity frame from the duplex callback. Only counts and contiguity are recorded here.</summary>
    public void OnIdentityFrame(in SpeakerCaptureFrame frame, bool renderActive, bool? nearEndAccepted)
    {
        Exception? failure = null;
        lock (_lock)
        {
            if (_completion is null)
                return;
            if (!frame.IsValid)
                return;

            CaptureRange range = frame.SourceRange;
            if (frame.HasDiscontinuity ||
                (_previousRange.HasValue && !_previousRange.Value.IsContiguousWith(in range)))
            {
                failure = new SpeakerEnrollmentSegmentException(
                    SpeakerEnrollmentReasonCode.TimelineGap,
                    "Speaker enrollment capture timeline was interrupted; the current segment is retried.");
            }
            else
            {
                _previousRange = range;
                _totalSamples += frame.Audio.Length;
                _frameCount++;
                if (renderActive)
                    _renderActiveFrames++;
                if (nearEndAccepted is { } accepted)
                {
                    _nearEndObserved = true;
                    if (accepted)
                        _nearEndAcceptedFrames++;
                }
            }
        }

        if (failure is not null)
            Fail(failure);
    }

    /// <summary>Speech chunk reported by the VAD for audio previously predicted from the identity tap.</summary>
    public void OnSpeech(ReadOnlySpan<float> speech)
    {
        if (speech.IsEmpty)
            return;

        TaskCompletionSource<SpeakerEnrollmentSegmentCapture>? completion = null;
        SpeakerEnrollmentSegmentCapture? completed = null;
        int collected;
        int required;
        bool report;
        lock (_lock)
        {
            if (_completion is null || _requiredSpeechSamples <= 0)
                return;

            int remaining = _requiredSpeechSamples - _speech.Count;
            int take = Math.Min(remaining, speech.Length);
            for (int i = 0; i < take; i++)
                _speech.Add(speech[i]);

            collected = _speech.Count;
            required = _requiredSpeechSamples;
            int percent = (int)((long)collected * 100 / required);
            report = percent != _lastReportedPercent;
            _lastReportedPercent = percent;

            if (collected >= required)
            {
                completion = _completion;
                _completion = null;
                completed = SnapshotLocked();
                ResetLocked();
            }
        }

        if (report)
            Progress?.Invoke(collected, required);
        if (completion is not null && completed is not null)
            completion.TrySetResult(completed);
    }

    private SpeakerEnrollmentSegmentCapture SnapshotLocked()
        => new(
            _speech.ToArray(),
            _totalSamples,
            _frameCount,
            _renderActiveFrames,
            _nearEndAcceptedFrames,
            _nearEndObserved);

    private void ResetLocked()
    {
        _completion = null;
        _speech.Clear();
        _previousRange = null;
        _totalSamples = 0;
        _frameCount = 0;
        _renderActiveFrames = 0;
        _nearEndAcceptedFrames = 0;
        _nearEndObserved = false;
        _requiredSpeechSamples = 0;
        _lastReportedPercent = -1;
    }
}
