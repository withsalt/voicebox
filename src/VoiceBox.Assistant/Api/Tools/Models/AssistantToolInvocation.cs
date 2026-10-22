using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;

namespace VoiceBox.Assistant.Api.Tools.Models
{
    public sealed class AssistantToolInvocation
    {
        internal AssistantToolInvocation(
            string callId,
            JsonElement arguments,
            AssistantToolExecutionPreference executionPreference)
        {
            CallId = callId;
            Arguments = arguments.Clone();
            ExecutionPreference = executionPreference;
        }

        public string CallId { get; }

        public JsonElement Arguments { get; }

        public AssistantToolExecutionPreference ExecutionPreference { get; }
    }
}
