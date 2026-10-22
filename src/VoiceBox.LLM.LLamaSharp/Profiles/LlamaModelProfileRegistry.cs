using System.Collections.Generic;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>兼容旧调用入口的静态注册表。</summary>
    public static class LlamaModelProfileRegistry
    {
        public static IReadOnlyList<ILlamaModelProfile> BuiltInProfiles =>
            BuiltInLlamaModelProfiles.All;

        public static ILlamaModelProfile Resolve(
            LlamaModelInfo info,
            ILlamaModelProfile? explicitProfile = null,
            IEnumerable<ILlamaModelProfile>? additionalProfiles = null)
        {
            return new AdaptiveLlamaModelProfileResolver(additionalProfiles)
                .Resolve(info, explicitProfile);
        }
    }
}
