using System.Buffers;
using System.Text;
using System.Text.Json;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.LLamaSharp.Internal;

internal static class LlamaChatContentFormatter
{
    public static string Format(ChatMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);

        if (message.Content.All(static content => content is TextContent))
        {
            var text = new StringBuilder();
            foreach (TextContent content in message.Content.Cast<TextContent>())
                text.Append(content.Text);
            return text.ToString();
        }

        ToolCallContent[] calls = message.Content.OfType<ToolCallContent>().ToArray();
        if (calls.Length > 0)
        {
            if (message.Role != ChatRole.Assistant || calls.Length != message.Content.Count)
                throw new NotSupportedException(
                    "LLamaSharp tool-call history must be an assistant message containing only tool calls.");
            return SerializeToolCalls(calls);
        }

        ToolResultContent[] results = message.Content.OfType<ToolResultContent>().ToArray();
        if (results.Length > 0)
        {
            if (message.Role != ChatRole.Tool || results.Length != 1 || message.Content.Count != 1)
                throw new NotSupportedException(
                    "LLamaSharp tool-result history must contain one tool result in a tool-role message.");
            return SerializeToolResult(results[0]);
        }

        if (message.Content.Any(static content => content is AudioContent or ImageContent))
        {
            throw new NotSupportedException(
                "VoiceBox.LLM.LLamaSharp does not support audio/image input yet " +
                "(llama.cpp multi-modal requires an mmproj model via mtmd). Use VoiceBox.LLM.OnnxGenAI or pass text only.");
        }

        throw new NotSupportedException(
            $"Unsupported chat content in LLamaSharp prompt: {string.Join(", ", message.Content.Select(static item => item.GetType().Name))}.");
    }

    private static string SerializeToolCalls(IReadOnlyList<ToolCallContent> calls) => Write(writer =>
    {
        writer.WriteStartObject();
        writer.WriteStartArray("tool_calls");
        foreach (ToolCallContent call in calls)
        {
            writer.WriteStartObject();
            writer.WriteString("id", call.CallId);
            writer.WriteString("name", call.ToolName);
            writer.WritePropertyName("arguments");
            using JsonDocument arguments = JsonDocument.Parse(call.ArgumentsJson);
            if (arguments.RootElement.ValueKind != JsonValueKind.Object)
                throw new FormatException("Tool-call history arguments must be a JSON object.");
            arguments.RootElement.WriteTo(writer);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WriteEndObject();
    });

    private static string SerializeToolResult(ToolResultContent result) => Write(writer =>
    {
        writer.WriteStartObject();
        writer.WriteStartObject("tool_result");
        writer.WriteString("call_id", result.CallId);
        writer.WriteString("result_json", result.ResultJson);
        writer.WriteBoolean("is_error", result.IsError);
        writer.WriteEndObject();
        writer.WriteEndObject();
    });

    private static string Write(Action<Utf8JsonWriter> write)
    {
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
            write(writer);
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
