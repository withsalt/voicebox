using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

namespace VoiceBox.Assistant.Conversation.Internal.Tools
{
    /// <summary>A finished call: the normalized result and the JSON the model receives for it.</summary>
    internal sealed record ToolOutcome(ConversationToolResult Result, string ModelResultJson)
    {
        /// <summary>Renders a result for the model. Throws <see cref="JsonException"/> when its structured content is not JSON.</summary>
        public static ToolOutcome From(ConversationToolResult result)
        {
            ArgumentNullException.ThrowIfNull(result);
            if (result.StructuredContentJson is { } structured)
                return new ToolOutcome(result, ModelJson.Canonicalize(structured));

            string json = ModelJson.Write(writer =>
            {
                writer.WriteStartObject();
                if (!result.IsSuccess)
                {
                    writer.WriteString("error", result.ErrorCode ?? result.Kind.ToString());
                    writer.WriteString("outcome", result.Kind.ToString());
                }
                if (result.IsSuccess || result.Content.Length > 0)
                    writer.WriteString("content", result.Content);
                writer.WriteEndObject();
            });
            return new ToolOutcome(result, json);
        }

        public static ToolOutcome Failure(ConversationToolResultKind kind, string errorCode, bool isTransient = false)
            => From(ConversationToolResult.Failure(kind, errorCode, isTransient: isTransient));
    }
}
