using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.Assistant.Modules.TurnBoundary.Adapters;

/// <summary>Adapts the current VAD provider to the typed turn-boundary port.</summary>
internal sealed class VadTurnBoundaryAdapter : ITurnBoundarySource
{
    private readonly IVoiceActivityDetectService _vad;
    private EventHandler<TurnBoundarySpeechEventArgs>? _speaking;
    private EventHandler<TurnBoundarySpeechEventArgs>? _speechEnded;

    public VadTurnBoundaryAdapter(IVoiceActivityDetectService vad)
        => _vad = vad ?? throw new ArgumentNullException(nameof(vad));

    public event EventHandler<EventArgs>? SpeechStarted
    {
        add => _vad.OnSpeechStarted += value;
        remove => _vad.OnSpeechStarted -= value;
    }

    public event EventHandler<TurnBoundarySpeechEventArgs>? Speaking
    {
        add
        {
            if (_speaking is null)
                _vad.OnSpeaking += OnVadSpeaking;
            _speaking += value;
        }
        remove
        {
            _speaking -= value;
            if (_speaking is null)
                _vad.OnSpeaking -= OnVadSpeaking;
        }
    }

    public event EventHandler<TurnBoundarySpeechEventArgs>? SpeechEnded
    {
        add
        {
            if (_speechEnded is null)
                _vad.OnSpeechEnded += OnVadSpeechEnded;
            _speechEnded += value;
        }
        remove
        {
            _speechEnded -= value;
            if (_speechEnded is null)
                _vad.OnSpeechEnded -= OnVadSpeechEnded;
        }
    }

    public void Reset() => _vad.Reset();

    public void Observe(ReadOnlySpan<float> audio) => _vad.Predict(audio);

    private void OnVadSpeaking(object? sender, VADEventArgs e)
        => _speaking?.Invoke(this, new TurnBoundarySpeechEventArgs(e.Data, e.Duration ?? TimeSpan.Zero));

    private void OnVadSpeechEnded(object? sender, VADEventArgs e)
        => _speechEnded?.Invoke(this, new TurnBoundarySpeechEventArgs(e.Data, e.Duration ?? TimeSpan.Zero));
}
