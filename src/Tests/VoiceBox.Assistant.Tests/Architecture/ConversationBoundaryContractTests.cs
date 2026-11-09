using System.Reflection;
using System.Text.RegularExpressions;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Wiring.Models;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Assistant.Kernel.Effects;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void AssistantReferencesTheConversationContractButNotConcreteImplementations()
    {
        Assembly assistant = typeof(VoiceAssistant).Assembly;
        Assert.Contains(
            assistant.GetReferencedAssemblies(),
            reference => reference.Name == "VoiceBox.Assistant.Conversation.Abstractions");
        Assert.DoesNotContain(
            assistant.GetReferencedAssemblies(),
            reference => reference.Name is
                "VoiceBox.Assistant.Conversation" or
                "VoiceBox.Assistant.Conversation.Mcp");

        string project = ReadSource("src", "VoiceBox.Assistant", "VoiceBox.Assistant.csproj");
        Assert.Contains("VoiceBox.Assistant.Conversation.Abstractions.csproj", project, StringComparison.Ordinal);
        Assert.DoesNotContain(
            "VoiceBox.Assistant.Conversation\\VoiceBox.Assistant.Conversation.csproj",
            project,
            StringComparison.Ordinal);
        Assert.DoesNotContain("Conversation.Mcp", project, StringComparison.Ordinal);
    }

    [Fact]
    public void AssistantConversationDependenciesStayWithinSessionAbstractions()
    {
        string assistantRoot = Path.Combine(FindRepositoryRoot(), "src", "VoiceBox.Assistant");
        string[] forbiddenNamespaces =
        [
            @"\bVoiceBox\.Assistant\.Conversation\b(?!\.Abstractions\b)",
            @"\bVoiceBox\.Assistant\.Conversation\.Abstractions\.Tools\b",
            @"\bVoiceBox\.Core\.Services\.LLMService\b",
            @"\bModelContextProtocol\b",
        ];

        string[] violations = EnumerateSourceFiles(assistantRoot)
            .SelectMany(path =>
            {
                string source = File.ReadAllText(path);
                return forbiddenNamespaces
                    .SelectMany(pattern => Regex.Matches(source, pattern))
                    .Select(match =>
                        $"{Path.GetRelativePath(assistantRoot, path)} references '{match.Value}'.");
            })
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.True(
            violations.Length == 0,
            "Assistant conversation-ownership violations:" +
            Environment.NewLine +
            string.Join(Environment.NewLine, violations));
    }

    [Fact]
    public void ConversationRegistrationAcceptsOnlyTheSessionContract()
    {
        MethodInfo registration = Assert.Single(typeof(VoiceAssistantBuilder).GetMethods(),
            method => method.Name == nameof(VoiceAssistantBuilder.UseConversation));
        Assert.Equal(typeof(IConversationSession), Assert.Single(registration.GetParameters()).ParameterType);
        Assert.Equal(typeof(VoiceAssistantBuilder), registration.ReturnType);
    }

    [Fact]
    public void AssistantEffectsStayWithinVoiceInteractionResponsibilities()
    {
        Assert.Equal(
            ["SetInputGate", "PrepareRecognition", "FinishRecognition", "GenerateResponse",
                "CancelResponse", "ClearPlaybackQueue", "StopPlayback", "PublishStatefulEvent",
                "CancelOperation", "CompleteTurn"],
            Enum.GetNames<AssistantEffectKind>());
        Assert.Equal(
            ["InputGate", "Recognition", "ResponseGeneration", "Playback", "PublicEventPublisher", "Runtime"],
            Enum.GetNames<AssistantEffectHandlerFamily>());
        Assert.Equal(
            ["AudioCapture", "AudioPlayback", "CaptureTimeline", "SimultaneousCapturePlayback",
                "RenderReference", "ProcessedCaptureTimeline", "AcousticProcessingDiagnostics",
                "CaptureProcessorStage", "SpeakerCaptureTimeline", "TurnBoundary", "SpeechRecognition",
                "ResponseGeneration", "SpeechSynthesis", "WakeupGate", "SpeakerEvidence", "BargeIn"],
            Enum.GetNames<VoiceBox.Assistant.Composition.Plans.AssistantCapability>());
    }
}
