using System;
using System.Collections.Generic;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>内置模型 Profile 目录；顺序即自动匹配优先级。</summary>
    internal static class BuiltInLlamaModelProfiles
    {
        public static IReadOnlyList<ILlamaModelProfile> All { get; } =
            Array.AsReadOnly<ILlamaModelProfile>(
            [
                new MiniCpm5LlamaModelProfile(),
                new Gemma4LlamaModelProfile(),
                new Llama3LlamaModelProfile(),
                new GemmaLlamaModelProfile(),
                new MistralLlamaModelProfile(),
                new Phi3LlamaModelProfile(),
                new Qwen35LlamaModelProfile(),
                new Qwen3LlamaModelProfile(),
                new ChatMlLlamaModelProfile(),
                new NativeLlamaModelProfile(),
                new RawLlamaModelProfile(),
            ]);
    }
}
