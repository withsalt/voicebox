using VoiceBox.ASR.Confucius4.Internal;

namespace VoiceBox.ASR.Confucius4.Internal;

internal interface IConfucius4Runtime : IDisposable
{
    string Rollback(string text, int count);
    string Decode(ReadOnlySpan<float> audio, string prefix, int maxTokens,
        IncrementalMelState melState, CancellationToken cancellationToken, bool final = false);
}
