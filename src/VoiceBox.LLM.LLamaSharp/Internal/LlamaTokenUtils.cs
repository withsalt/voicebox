using System.Collections.Generic;
using LLama.Native;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    internal static class LlamaTokenUtils
    {
        /// <summary>两个 token 序列的最长公共前缀长度 (会话 KV-cache 复用的对账依据)。</summary>
        public static int CommonPrefixLength(IReadOnlyList<LLamaToken> a, IReadOnlyList<LLamaToken> b)
        {
            int max = a.Count < b.Count ? a.Count : b.Count;
            int i = 0;
            while (i < max && a[i] == b[i])
                i++;
            return i;
        }
    }
}
