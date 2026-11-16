using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// Owns the Runtime subscriptions. It is intentionally a closed, typed surface rather than a
/// general event registry, and it only binds the optional timeline sources the frontend declares.
/// The duplex frame subscription is taken once; the router fans it out to the identity tap and the
/// render-aware detector.
/// </summary>
internal sealed class AssistantRuntimeSubscriptionOwner
{
    private readonly object _lock = new();
    private readonly IProcessedCaptureSource? _processedCaptureSource;
    private readonly IDuplexAudioFrameSource? _duplexFrameSource;
    private readonly ITurnBoundarySource _turnBoundary;
    private readonly ProcessedCaptureFrameHandler _processedCapture;
    private readonly DuplexAudioFrameHandler _duplexFrame;
    private readonly EventHandler<EventArgs> _speechStarted;
    private readonly EventHandler<TurnBoundarySpeechEventArgs> _speaking;
    private readonly EventHandler<TurnBoundarySpeechEventArgs> _speechEnded;
    private bool _isAttached;

    public AssistantRuntimeSubscriptionOwner(
        IProcessedCaptureSource? processedCaptureSource,
        IDuplexAudioFrameSource? duplexFrameSource,
        ITurnBoundarySource turnBoundary,
        ProcessedCaptureFrameHandler processedCapture,
        DuplexAudioFrameHandler duplexFrame,
        EventHandler<EventArgs> speechStarted,
        EventHandler<TurnBoundarySpeechEventArgs> speaking,
        EventHandler<TurnBoundarySpeechEventArgs> speechEnded)
    {
        _processedCaptureSource = processedCaptureSource;
        _duplexFrameSource = duplexFrameSource;
        _turnBoundary = turnBoundary ?? throw new ArgumentNullException(nameof(turnBoundary));
        _processedCapture = processedCapture ?? throw new ArgumentNullException(nameof(processedCapture));
        _duplexFrame = duplexFrame ?? throw new ArgumentNullException(nameof(duplexFrame));
        _speechStarted = speechStarted ?? throw new ArgumentNullException(nameof(speechStarted));
        _speaking = speaking ?? throw new ArgumentNullException(nameof(speaking));
        _speechEnded = speechEnded ?? throw new ArgumentNullException(nameof(speechEnded));
    }

    public bool IsAttached
    {
        get
        {
            lock (_lock)
                return _isAttached;
        }
    }

    public void Attach()
    {
        lock (_lock)
        {
            if (_isAttached)
                return;

            if (_processedCaptureSource is not null)
                _processedCaptureSource.OnProcessedCaptureFrame += _processedCapture;
            if (_duplexFrameSource is not null)
                _duplexFrameSource.OnDuplexAudioFrame += _duplexFrame;
            _turnBoundary.SpeechStarted += _speechStarted;
            _turnBoundary.Speaking += _speaking;
            _turnBoundary.SpeechEnded += _speechEnded;
            _isAttached = true;
        }
    }

    public void Detach()
    {
        lock (_lock)
        {
            if (!_isAttached)
                return;

            if (_processedCaptureSource is not null)
                _processedCaptureSource.OnProcessedCaptureFrame -= _processedCapture;
            if (_duplexFrameSource is not null)
                _duplexFrameSource.OnDuplexAudioFrame -= _duplexFrame;
            _turnBoundary.SpeechStarted -= _speechStarted;
            _turnBoundary.Speaking -= _speaking;
            _turnBoundary.SpeechEnded -= _speechEnded;
            _isAttached = false;
        }
    }
}
