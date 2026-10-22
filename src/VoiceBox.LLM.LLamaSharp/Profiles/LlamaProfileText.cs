using System;

namespace VoiceBox.LLM.LLamaSharp
{
    internal static class LlamaProfileText
    {
        public static bool Contains(string? source, string value) =>
            source?.Contains(value, StringComparison.OrdinalIgnoreCase) ?? false;
    }
}
