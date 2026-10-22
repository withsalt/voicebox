using VoiceBox.Assistant.Modules.ModelInteraction.Models;

namespace VoiceBox.Assistant.Modules.ModelInteraction.Contracts;

internal interface IAssistantToolRuntimeFactory
{
    /// <summary>
    /// Creates the runtime for one assistant instance. Host policy that the runtime boundary must
    /// enforce is passed in at creation time, so registration stays free of assistant options and a
    /// validated composition remains inert until an instance is assembled.
    /// </summary>
    IAssistantToolRuntime Create(AssistantToolRuntimePolicy policy);
}
