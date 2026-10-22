using System;
using System.Collections.Generic;
using LLama;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>
    /// 扁平化后的一条对话消息 (角色字符串 + 纯文本), 供 chat template 渲染。
    /// </summary>
    public readonly record struct LlamaChatMessage(string Role, string Text);

    /// <summary>
    /// 从 GGUF 元数据提炼出的模型标识信息, 供 profile 匹配与诊断日志使用。
    /// </summary>
    public sealed class LlamaModelInfo
    {
        public LlamaModelInfo(IReadOnlyDictionary<string, string> metadata)
        {
            ArgumentNullException.ThrowIfNull(metadata);
            Metadata = metadata;
            Architecture = metadata.GetValueOrDefault("general.architecture", string.Empty);
            Name = metadata.GetValueOrDefault("general.name", string.Empty);
            ChatTemplate = metadata.GetValueOrDefault("tokenizer.chat_template");
        }

        /// <summary>模型架构 (GGUF <c>general.architecture</c>), 例如 "llama" / "gemma3" / "minicpm"。</summary>
        public string Architecture { get; }

        /// <summary>模型名 (GGUF <c>general.name</c>)。</summary>
        public string Name { get; }

        /// <summary>GGUF 内嵌的 chat template (jinja 源码); 无内嵌模板时为 null。</summary>
        public string? ChatTemplate { get; }

        /// <summary>完整 GGUF 元数据 (key → value 字符串)。</summary>
        public IReadOnlyDictionary<string, string> Metadata { get; }
    }

    /// <summary>
    /// 模型差异扩展点 (策略模式)。不同 GGUF 模型在 chat template、thinking 标记、
    /// 停止词上各有差异; 服务按 GGUF 元数据自动匹配内置 profile, 特殊模型可通过
    /// <see cref="LLamaSharpLLMOptions.ModelProfile"/> 注入自定义实现 (或继承
    /// <see cref="DefaultLlamaModelProfile"/> 只覆盖差异部分), 无需修改库代码。
    /// </summary>
    public interface ILlamaModelProfile
    {
        /// <summary>profile 名称 (诊断日志用)。</summary>
        string Name { get; }

        /// <summary>本 profile 是否适用于给定模型。仅在自动匹配时被调用。</summary>
        bool Matches(LlamaModelInfo info);

        /// <summary>
        /// 把扁平化消息渲染为最终 prompt 字符串。
        /// <paramref name="enableThinking"/> 透传自 <c>LlmCompleteOptions.EnableThinking</c>,
        /// 由 profile 决定如何落实 (默认实现忽略: llama.cpp 内置模板引擎不支持该开关,
        /// 模型自发的 thinking 输出仍会被标记路由到 Thinking 块)。
        /// </summary>
        string BuildPrompt(
            LLamaWeights weights,
            IReadOnlyList<LlamaChatMessage> messages,
            bool addGenerationPrompt,
            bool enableThinking);

        /// <summary>
        /// 生成开始时是否已经处于 thinking 块内。部分模型会把 thinking 起始标记预填进
        /// prompt，因而解码器看不到该标记；profile 必须显式告知流式解码器初始状态。
        /// 默认实现保持向后兼容，适用于由模型自行生成 thinking 起始标记的模板。
        /// </summary>
        bool StartsInsideThinking(bool enableThinking) => false;

        /// <summary>thinking 块起始标记 (文本形态, 如 <c>&lt;think&gt;</c>)。</summary>
        IReadOnlyList<string> ThinkingStartMarkers { get; }

        /// <summary>thinking 块结束标记。</summary>
        IReadOnlyList<string> ThinkingEndMarkers { get; }

        /// <summary>只需从输出中剔除、不改变解码状态的标记 (泄漏出的 special token 文本)。</summary>
        IReadOnlyList<string> StripMarkers { get; }

        /// <summary>
        /// 文本级停止词 (anti-prompt)。命中即结束生成, 停止词本身不进入输出。
        /// 绝大多数模型靠 EOG token 停机, 这里仅作 GGUF 模板/EOG 元数据异常时的补充手段。
        /// </summary>
        IReadOnlyList<string> StopTexts { get; }
    }

}
