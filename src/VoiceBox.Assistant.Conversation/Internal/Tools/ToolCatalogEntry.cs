using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.Assistant.Conversation.Internal.Tools
{
    /// <summary>One tool of a started run, with everything derived from its definition computed once.</summary>
    internal sealed class ToolCatalogEntry
    {
        public ToolCatalogEntry(ConversationToolDefinition definition, string exposedName)
        {
            Definition = definition;
            ExposedName = exposedName;
            // Canonical property order keeps the provider request byte-stable across runs.
            string modelSchema = ToolExecutionPreferenceSchema.AddTo(
                ModelJson.Canonicalize(definition.Descriptor.InputSchemaJson),
                definition.Descriptor.ExecutionPolicy);
            ModelDefinition = new LlmToolDefinition(exposedName, definition.Descriptor.Description, modelSchema, strict: false);
        }

        public ConversationToolDefinition Definition { get; }

        public ConversationToolId Id => Definition.Id;

        public ConversationToolDescriptor Descriptor => Definition.Descriptor;

        /// <summary>Provider-safe function name the model sees and calls.</summary>
        public string ExposedName { get; }

        /// <summary>Function definition sent to the provider, including the execution preference property.</summary>
        public LlmToolDefinition ModelDefinition { get; }
    }
}
