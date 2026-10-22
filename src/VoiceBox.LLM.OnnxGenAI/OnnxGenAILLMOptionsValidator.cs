using System;
using Microsoft.Extensions.Options;

namespace VoiceBox.LLM.OnnxGenAI
{
    /// <summary>
    /// <see cref="OnnxGenAILLMOptions"/> 的唯一校验入口。
    ///
    /// 服务构造 (<see cref="OnnxGenAILLMService"/> 直接 new) 与 DI 注册两条路径共用同一套规则,
    /// 避免两处各维护一份而漂移。这里只做与运行环境无关的不变式校验;
    /// 目录/genai_config.json 是否存在由 <see cref="Internal.GenAIInferenceSession"/> 负责 (那里能给出更精确的报错)。
    /// </summary>
    internal sealed class OnnxGenAILLMOptionsValidator : IValidateOptions<OnnxGenAILLMOptions>
    {
        public ValidateOptionsResult Validate(string? name, OnnxGenAILLMOptions options)
        {
            string? error = Validate(options);
            return error is null ? ValidateOptionsResult.Success : ValidateOptionsResult.Fail(error);
        }

        /// <summary>返回第一条校验错误信息; <c>null</c> 表示通过。</summary>
        public static string? Validate(OnnxGenAILLMOptions options)
        {
            if (options is null) return "Options cannot be null.";
            if (string.IsNullOrWhiteSpace(options.ModelPath)) return "ModelPath cannot be null or empty.";
            if (options.MaxConcurrentRequests < 1) return "MaxConcurrentRequests must be >= 1.";
            if (options.MaxTokens < 1) return "MaxTokens must be >= 1.";
            if (options.StreamChannelCapacity < 0) return "StreamChannelCapacity must be >= 0.";
            if (options.GpuDeviceId < 0) return "GpuDeviceId must be >= 0.";
            if (options.MaxLengthMediaSafetyMargin < 0) return "MaxLengthMediaSafetyMargin must be >= 0.";
            if (options.SessionMaxLength < 1) return "SessionMaxLength must be >= 1.";
            return null;
        }

        /// <summary>校验失败时抛 <see cref="ArgumentException"/>; 供直接构造服务的路径使用。</summary>
        public static void ThrowIfInvalid(OnnxGenAILLMOptions options)
        {
            string? error = Validate(options);
            if (error is not null)
                throw new ArgumentException(error, nameof(options));
        }
    }
}
