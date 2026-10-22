using System;
using System.Collections.Generic;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>
    /// 可扩展的自适应模型兼容解析器。解析顺序为：强制 profile、调用方附加 profile、
    /// 内置特征适配器、GGUF 原生模板、Raw 最终兜底。首个 <see cref="ILlamaModelProfile.Matches"/>
    /// 返回 true 的适配器获选，因此自定义适配器可以安全覆盖内置行为。
    /// </summary>
    public sealed class AdaptiveLlamaModelProfileResolver
    {
        private readonly IReadOnlyList<ILlamaModelProfile> _profiles;

        public AdaptiveLlamaModelProfileResolver(IEnumerable<ILlamaModelProfile>? additionalProfiles = null)
        {
            var profiles = new List<ILlamaModelProfile>();
            if (additionalProfiles is not null)
            {
                foreach (var profile in additionalProfiles)
                {
                    if (profile is null)
                        throw new ArgumentException(
                            "Model profile collection cannot contain null.",
                            nameof(additionalProfiles));
                    profiles.Add(profile);
                }
            }

            profiles.AddRange(BuiltInLlamaModelProfiles.All);
            _profiles = profiles.AsReadOnly();
        }

        /// <summary>按实际匹配顺序排列的全部适配器。</summary>
        public IReadOnlyList<ILlamaModelProfile> Profiles => _profiles;

        /// <summary>库内置适配器，按从具体到通用的顺序排列。</summary>
        public static IReadOnlyList<ILlamaModelProfile> BuiltInProfiles =>
            BuiltInLlamaModelProfiles.All;

        public ILlamaModelProfile Resolve(
            LlamaModelInfo info,
            ILlamaModelProfile? explicitProfile = null)
        {
            ArgumentNullException.ThrowIfNull(info);
            if (explicitProfile is not null)
                return explicitProfile;

            foreach (var profile in _profiles)
            {
                if (profile.Matches(info))
                    return profile;
            }

            // 内置列表末尾永远有 Raw；保留明确异常以防未来注册表被错误修改。
            throw new InvalidOperationException("No compatible llama prompt profile was found.");
        }
    }
}
