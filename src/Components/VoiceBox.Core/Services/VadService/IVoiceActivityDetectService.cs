using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.Core.Services.VadService
{
    public interface IVoiceActivityDetectService : IDisposable
    {
        event EventHandler<VADEventArgs>? OnSpeechEnded;

        event EventHandler<EventArgs>? OnSpeechStarted;

        event EventHandler<VADEventArgs>? OnSpeaking;

        event EventHandler<VadProbabilityEventArgs>? OnProbability;

        void Predict(ReadOnlySpan<float> audioData);

        void Reset();
    }
}
