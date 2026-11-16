using MiniAudioEx.Native;

namespace VoiceBox.Audio.MiniAudio.Backend;

internal static class MiniAudioNativeRuntime
{
    internal static readonly SemaphoreSlim ContextGate = new(1, 1);

    internal static void ThrowIfFailed(ma_result result, string operation)
    {
        if (result != ma_result.success)
        {
            throw new InvalidOperationException($"MiniAudio {operation} failed: {result}.");
        }
    }
}
