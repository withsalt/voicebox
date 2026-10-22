using System;
using System.Collections.Generic;
using System.Text;

namespace VoiceBox.Assistant.Api.Tools.Models
{
    public sealed record AssistantToolApprovalContext(string ServerAlias, AssistantToolDescriptor Tool, AssistantToolInvocation Invocation, string ArgumentsHash);
}
