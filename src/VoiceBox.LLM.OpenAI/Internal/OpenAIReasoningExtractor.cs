using System;
using System.ClientModel.Primitives;
using System.Text.Json;
using OpenAI.Chat;

namespace VoiceBox.LLM.OpenAI.Internal
{
    /// <summary>
    /// 从流式增量中提取推理内容 (<c>reasoning_content</c>)。
    ///
    /// 背景: <c>reasoning_content</c> 是 DeepSeek-R1 / deepseek-reasoner 等推理模型的非标准扩展字段,
    /// 官方 OpenAI SDK 的强类型 <see cref="StreamingChatCompletionUpdate"/> 不直接暴露它。但
    /// System.ClientModel 生成的模型会把未知 JSON 字段原样保留并在序列化时回写, 因此我们把整块
    /// update 序列化回 JSON, 再从 <c>choices[0].delta.reasoning_content</c> 取值 —— 既不绕过 SDK,
    /// 又能拿到扩展字段。
    ///
    /// 对不返回该字段的普通模型, 这里始终返回 <c>null</c>, 是无害的空操作。任何异常都被吞掉并返回
    /// <c>null</c>: 推理内容是锦上添花, 不能因其解析失败而中断正常文本流。
    /// </summary>
    internal static class OpenAIReasoningExtractor
    {
        public static string? TryExtract(StreamingChatCompletionUpdate update)
        {
            try
            {
                BinaryData json = ModelReaderWriter.Write(update);
                using JsonDocument doc = JsonDocument.Parse(json);

                if (!doc.RootElement.TryGetProperty("choices", out var choices)
                    || choices.ValueKind != JsonValueKind.Array
                    || choices.GetArrayLength() == 0)
                    return null;

                if (!choices[0].TryGetProperty("delta", out var delta)
                    || delta.ValueKind != JsonValueKind.Object)
                    return null;

                // 主流命名是 reasoning_content; 个别网关用 reasoning, 一并兼容。
                if (TryGetNonEmptyString(delta, "reasoning_content", out var reasoning)
                    || TryGetNonEmptyString(delta, "reasoning", out reasoning))
                    return reasoning;

                return null;
            }
            catch
            {
                return null;
            }
        }

        private static bool TryGetNonEmptyString(JsonElement obj, string name, out string? value)
        {
            value = null;
            if (obj.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.String)
            {
                string? s = prop.GetString();
                if (!string.IsNullOrEmpty(s))
                {
                    value = s;
                    return true;
                }
            }
            return false;
        }
    }
}
