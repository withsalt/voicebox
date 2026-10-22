using System;

namespace VoiceBox.LLM.OnnxGenAI
{
    /// <summary>
    /// token 序列相关的纯工具函数 (会话跨轮 KV-cache 复用的核心比对逻辑)。
    /// </summary>
    internal static class GenAITokenUtils
    {
        /// <summary>
        /// 两个 token 序列的最长公共前缀长度。会话据此决定 <c>RewindTo</c> 的边界 (复用多少既有前缀)。
        /// </summary>
        public static int CommonPrefixLength(ReadOnlySpan<int> a, ReadOnlySpan<int> b) =>
            a.CommonPrefixLength(b);
    }
}
