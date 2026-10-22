using System;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.OpenAI
{
    /// <summary>
    /// OpenAI 协议 LLM 服务配置。
    ///
    /// 适配任意"OpenAI 兼容"端点 —— 官方 OpenAI、Azure OpenAI、以及国内火山方舟(Ark)、
    /// DeepSeek、Moonshot、阿里百炼等。它们都暴露 <c>/chat/completions</c> 协议, 只是
    /// <see cref="Endpoint"/> / <see cref="Model"/> / <see cref="ApiKey"/> 不同。
    ///
    /// 与本地 ONNX Runtime GenAI 实现 (<c>VoiceBox.LLM.OnnxGenAI</c>) 不同, 这里是远程 HTTP 服务 —— 真正的
    /// 模型加载、KV-cache、GPU 调度都在服务端, 客户端只是 HTTP/JSON 桥。所以这里的选项以
    /// "连接 / 采样语义" 为主, 不涉及执行提供者或显存配额。
    /// </summary>
    public sealed class OpenAILLMOptions
    {
        /// <summary>
        /// 访问凭据 (Bearer token)。多数 OpenAI 兼容服务都要求, 留空时大概率以 401 失败。
        /// </summary>
        public string ApiKey { get; set; } = string.Empty;

        /// <summary>
        /// 服务根 URI, 即官方 <c>base_url</c>。SDK 会在其后追加 <c>/chat/completions</c>。
        /// <list type="bullet">
        ///   <item>官方 OpenAI: <c>https://api.openai.com/v1</c></item>
        ///   <item>火山方舟 Ark: <c>https://ark.cn-beijing.volces.com/api/v3</c></item>
        ///   <item>DeepSeek: <c>https://api.deepseek.com/v1</c></item>
        /// </list>
        /// 注意: 末尾不要带 <c>/chat/completions</c>, 否则会拼成双重路径。
        /// </summary>
        public Uri Endpoint { get; set; } = new Uri("https://api.openai.com/v1");

        /// <summary>
        /// 模型标识, 例如 <c>gpt-4o-mini</c> / <c>deepseek-v4-flash-260425</c>。
        /// 必须是目标服务实际提供的模型名, 否则首次请求会以 404 / 400 失败。
        /// </summary>
        public string Model { get; set; } = string.Empty;

        /// <summary>
        /// 单次网络往返超时 (映射到 SDK 管线的 <c>NetworkTimeout</c>)。
        /// 长文本生成可能持续数分钟, 默认给 5 分钟; 更长任务请显式调高。
        /// </summary>
        public TimeSpan HttpTimeout { get; set; } = TimeSpan.FromMinutes(5);

        /// <summary>
        /// 客户端侧的最大并发请求数。远程服务通常能并行, 但仍保留一道反压闸门, 避免触发
        /// 上游限流 (HTTP 429)。默认 4; 按账户配额酌情调整。
        /// </summary>
        public int MaxConcurrentRequests { get; set; } = 4;

        /// <summary>
        /// 默认 <b>新生成</b> token 上限 (映射到 <c>max_tokens</c> / <c>max_completion_tokens</c>)。
        /// 可由每次调用的 <see cref="LlmCompleteOptions.MaxTokens"/> 覆盖。
        /// 设为 0 或负数则不下发该参数 (沿用服务端默认)。
        /// </summary>
        public int MaxTokens { get; set; } = 4096;

        /// <summary>
        /// 是否启用采样。<c>false</c> 表示近似贪心解码; 内部通过把 <c>temperature</c> 设为 0 实现。
        /// </summary>
        public bool DoSample { get; set; } = true;

        /// <summary>
        /// 采样温度 (仅 <see cref="DoSample"/>=true 时生效)。OpenAI 协议取值区间通常为 [0, 2]。
        /// </summary>
        public float Temperature { get; set; } = 0.7f;

        /// <summary>
        /// Top-P (nucleus) 采样。1.0 表示不裁剪。
        /// </summary>
        public float TopP { get; set; } = 0.9f;

        /// <summary>
        /// 频率惩罚 (<c>frequency_penalty</c>), 取值 [-2, 2]。<c>null</c> 表示不下发。
        /// 注意: OpenAI 协议没有 <c>repetition_penalty</c>, 抑制重复请用本字段或
        /// <see cref="PresencePenalty"/>。
        /// </summary>
        public float? FrequencyPenalty { get; set; }

        /// <summary>
        /// 存在惩罚 (<c>presence_penalty</c>), 取值 [-2, 2]。<c>null</c> 表示不下发。
        /// </summary>
        public float? PresencePenalty { get; set; }

        /// <summary>
        /// 随机种子。<c>null</c> 表示由服务端自行选择 (非确定性); 设值则尽量复现
        /// (是否真正确定取决于服务端实现)。
        /// </summary>
        public int? Seed { get; set; }

        /// <summary>
        /// 深度思考开关 (请求字段 <c>thinking.type</c>)。<c>null</c> 表示不下发该参数, 沿用服务端默认。
        /// <see cref="ThinkingMode.Enabled"/> / <see cref="ThinkingMode.Disabled"/> 会分别下发
        /// <c>enabled</c> / <c>disabled</c>；<see cref="ThinkingMode.Auto"/> 不下发该字段。
        /// </summary>
        public ThinkingMode? Thinking { get; set; }

        /// <summary>
        /// 推理/思维链长度 (请求字段 <c>reasoning_effort</c>)。
        /// <c>null</c> 表示不下发该参数, 沿用服务端默认。火山方舟支持 <c>minimal</c> / <c>low</c> /
        /// <c>medium</c> / <c>high</c>, 部分模型支持 <c>max</c>。
        /// </summary>
        public OpenAIReasoningEffortLevel? ReasoningEffort { get; set; }

        /// <summary>
        /// 兼容旧配置的思考开关。<c>null</c> 表示不下发; <c>false</c> 映射为
        /// <c>thinking={"type":"disabled"}</c>, <c>true</c> 映射为 <c>thinking={"type":"enabled"}</c>。
        /// 新代码优先使用 <see cref="Thinking"/>。
        /// </summary>
        public bool? EnableThinking { get; set; }

        /// <summary>
        /// 停止序列。匹配到任一字符串即停止生成。<c>null</c> 表示不下发。
        /// </summary>
        public string[]? StopSequences { get; set; }

        /// <summary>
        /// 是否把推理模型 (如 deepseek-reasoner) 返回的 <c>reasoning_content</c> 增量作为
        /// <see cref="LlmChunkKind.Thinking"/> 单独路由。对不返回该字段的普通模型 (如
        /// deepseek-v4-flash) 是无害的空操作。默认开启。
        /// </summary>
        public bool SurfaceReasoning { get; set; } = true;

        /// <summary>
        /// 流式输出时 Channel 的有界容量。生成速度远快于消费时用此值反压。
        /// 默认 64 块基本不会成为瓶颈; 设为 0 表示无界 (不推荐, 可能 OOM)。
        /// </summary>
        public int StreamChannelCapacity { get; set; } = 64;
    }
}
