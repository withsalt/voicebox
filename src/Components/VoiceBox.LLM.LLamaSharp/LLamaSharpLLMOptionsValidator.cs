using System;
using Microsoft.Extensions.Options;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>
    /// <see cref="LLamaSharpLLMOptions"/> 的唯一校验入口。
    ///
    /// 服务构造 (<see cref="LLamaSharpLLMService"/> 直接 new) 与 DI 注册两条路径共用同一套规则,
    /// 避免两处各维护一份而漂移。这里只做与运行环境无关的不变式校验;
    /// 模型文件是否存在/是否为合法 GGUF 由 <see cref="Internal.LlamaModelSession"/> 负责
    /// (那里能给出更精确的报错)。
    /// </summary>
    internal sealed class LLamaSharpLLMOptionsValidator : IValidateOptions<LLamaSharpLLMOptions>
    {
        public ValidateOptionsResult Validate(string? name, LLamaSharpLLMOptions options)
        {
            string? error = Validate(options);
            return error is null ? ValidateOptionsResult.Success : ValidateOptionsResult.Fail(error);
        }

        /// <summary>返回第一条校验错误信息; <c>null</c> 表示通过。</summary>
        public static string? Validate(LLamaSharpLLMOptions options)
        {
            if (options is null) return "Options cannot be null.";
            if (string.IsNullOrWhiteSpace(options.ModelPath)) return "ModelPath cannot be null or empty.";
            if (options.ContextSize is < 8) return "ContextSize must be >= 8 (or null to follow the GGUF).";
            if (options.GpuLayerCount < -1) return "GpuLayerCount must be >= -1.";
            if (options.MainGpu < 0) return "MainGpu must be >= 0.";
            if (options.Threads is < 1) return "Threads must be >= 1 (or null for auto).";
            if (options.BatchSize < 1) return "BatchSize must be >= 1.";
            if (!Enum.IsDefined(options.LoadMode)) return "LoadMode must be a defined LLamaLoadMode value.";
            // llama.cpp context 不是线程安全的: 无状态推理路径复用单个 SharedContext,
            // MaxConcurrentRequests > 1 会导致多个请求并发读写同一 context 的 KV-cache。
            // 多用户场景请通过 IConversationalLLMService.CreateConversation() 实现
            // (每段会话持有独立 context, 各自串行)。
            if (options.MaxConcurrentRequests != 1) return "MaxConcurrentRequests must be 1. " +
                "The stateless path reuses a single LLamaContext which is not thread-safe. " +
                "For concurrent multi-user scenarios, use CreateConversation() — each conversation holds its own context.";
            if (options.MaxTokens < 1) return "MaxTokens must be >= 1.";
            if (options.Temperature < 0) return "Temperature must be >= 0.";
            if (options.TopK < 0) return "TopK must be >= 0.";
            if (options.TopP is <= 0 or > 1) return "TopP must be in (0, 1].";
            if (options.RepetitionPenalty <= 0) return "RepetitionPenalty must be > 0.";
            if (options.RepetitionPenaltyWindow < 0) return "RepetitionPenaltyWindow must be >= 0.";
            if (options.RandomSeed is < 0) return "RandomSeed must be >= 0 (or null for non-deterministic).";
            if (options.StreamChannelCapacity < 0) return "StreamChannelCapacity must be >= 0.";
            return null;
        }

        /// <summary>校验失败时抛 <see cref="ArgumentException"/>; 供直接构造服务的路径使用。</summary>
        public static void ThrowIfInvalid(LLamaSharpLLMOptions options)
        {
            string? error = Validate(options);
            if (error is not null)
                throw new ArgumentException(error, nameof(options));
        }
    }
}
