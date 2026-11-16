using System;
using System.Collections.Generic;
using System.Text;
using LLama;
using LLama.Native;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    internal static class LlamaTokenUtils
    {
        /// <summary>模板已带 BOS 时先移除它，再由分词器按模型配置统一添加 BOS/EOS，避免重复。</summary>
        public static LLamaToken[] TokenizePrompt(LLamaWeights weights, string prompt, Encoding encoding)
        {
            var vocab = weights.Vocab;
            if (vocab.ShouldAddBOS)
            {
                string? bos = vocab.LLamaTokenToString(vocab.BOS, isSpecialToken: true);
                if (!string.IsNullOrEmpty(bos) && prompt.StartsWith(bos, StringComparison.Ordinal))
                    prompt = prompt[bos.Length..];
            }

            // addBos 实际控制 native 的 add_special；保持开启以保留模型要求的 EOS。
            return weights.Tokenize(prompt, add_bos: true, special: true, encoding: encoding);
        }

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
