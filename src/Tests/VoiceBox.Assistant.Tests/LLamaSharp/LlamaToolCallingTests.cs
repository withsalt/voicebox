using System.Text.Json;
using LLama.Sampling;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.LLamaSharp;
using VoiceBox.LLM.LLamaSharp.Internal;
using Xunit;

namespace VoiceBox.Assistant.Tests.LLamaSharp;

public sealed class LlamaToolCallingTests
{
    private const string Schema =
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}," +
        "\"required\":[\"city\"],\"additionalProperties\":false}";

    [Fact]
    public void CatalogBuildsBoundedEnvelopeGrammarAndIncludesToolSchemaInPrompt()
    {
        LlamaToolCallCatalog catalog = CreateCatalog();

        Assert.Contains("root ::= final | tool-request", catalog.Grammar, StringComparison.Ordinal);
        Assert.Contains("tool-call-list-16", catalog.Grammar, StringComparison.Ordinal);
        Assert.Contains("json-object-0", catalog.Grammar, StringComparison.Ordinal);
        Assert.Contains("local__weather", catalog.Grammar, StringComparison.Ordinal);
        Assert.Contains(Schema, catalog.SystemPrompt, StringComparison.Ordinal);
    }

    [Fact]
    public void CatalogRejectsSchemasThatCannotDescribeObjectArguments()
    {
        var definition = new LlmToolDefinition(
            "local__weather",
            "Gets current weather for one city.",
            "{\"type\":\"string\"}");

        Assert.Throws<ArgumentException>(() => LlamaToolCallCatalog.Create([definition]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GrammarIsAttachedToBothGreedyAndSampledPipelines(bool doSample)
    {
        const string grammarText = "root ::= \"ok\"";
        var sampling = new LlamaSamplingOptions(
            doSample,
            Temperature: 0.7f,
            TopK: 40,
            TopP: 0.9f,
            RepetitionPenalty: 1.0f,
            PenaltyWindow: 64,
            Seed: null);

        using ISamplingPipeline pipeline = sampling.CreatePipeline(grammarText);
        Grammar? grammar = pipeline switch
        {
            GreedySamplingPipeline greedy => greedy.Grammar,
            DefaultSamplingPipeline sampled => sampled.Grammar,
            _ => null,
        };

        Assert.NotNull(grammar);
        Assert.Equal(grammarText, grammar.Gbnf);
        Assert.Equal("root", grammar.Root);
    }

    [Fact]
    public void ParserSeparatesFinalTextAndValidatedToolCalls()
    {
        LlamaToolCallCatalog catalog = CreateCatalog();

        var final = Assert.IsType<LlamaToolCallOutcome.Final>(
            LlamaToolCallEnvelopeParser.Parse("{\"text\":\"It is sunny.\"}", catalog));
        Assert.Equal("It is sunny.", final.Text);

        var calls = Assert.IsType<LlamaToolCallOutcome.ToolCalls>(LlamaToolCallEnvelopeParser.Parse(
            "{\"tool_calls\":[{\"name\":\"local__weather\",\"arguments\":{\"city\":\"杭州\"}}]}",
            catalog));
        LlamaToolCall call = Assert.Single(calls.Calls);
        Assert.Equal("local__weather", call.ToolName);
        using JsonDocument arguments = JsonDocument.Parse(call.ArgumentsJson);
        Assert.Equal("杭州", arguments.RootElement.GetProperty("city").GetString());
    }

    [Theory]
    [InlineData("{\"text\":\"ok\",\"extra\":true}")]
    [InlineData("{\"tool_calls\":[{\"name\":\"unknown\",\"arguments\":{}}]}")]
    [InlineData("{\"tool_calls\":[{\"name\":\"local__weather\",\"arguments\":[]}]}")]
    [InlineData("{\"tool_calls\":[{\"name\":\"local__weather\",\"arguments\":{\"city\":\"x\",\"city\":\"y\"}}]}")]
    public void ParserRejectsInvalidOrAmbiguousResponses(string output)
    {
        Assert.Throws<FormatException>(() =>
            LlamaToolCallEnvelopeParser.Parse(output, CreateCatalog()));
    }

    [Fact]
    public void ToolPromptPreservesSystemPolicyAndSerializesCorrelatedToolHistory()
    {
        LlamaToolCallCatalog catalog = CreateCatalog();
        IReadOnlyList<ChatMessage> prompt = LlamaToolPromptBuilder.AddToolInstructions(
        [
            ChatMessage.System("Answer in Chinese."),
            ChatMessage.User("What is the weather in Hangzhou?"),
            ChatMessage.AssistantToolCalls(
                new ToolCallContent("call-1", "local__weather", "{\"city\":\"杭州\"}")),
            ChatMessage.ToolResult("call-1", "{\"temperature\":25}"),
        ],
        catalog);

        Assert.Equal(ChatRole.System, prompt[0].Role);
        string systemText = Assert.IsType<TextContent>(prompt[0].Content[0]).Text;
        Assert.Contains("Answer in Chinese.", systemText, StringComparison.Ordinal);
        Assert.Contains(Schema, systemText, StringComparison.Ordinal);

        List<LlamaChatMessage> flattened = LlamaPromptBuilder.Flatten(prompt);
        Assert.Equal("assistant", flattened[2].Role);
        using JsonDocument callHistory = JsonDocument.Parse(flattened[2].Text);
        Assert.Equal("call-1", callHistory.RootElement
            .GetProperty("tool_calls")[0].GetProperty("id").GetString());
        Assert.Equal("tool", flattened[3].Role);
        using JsonDocument resultHistory = JsonDocument.Parse(flattened[3].Text);
        Assert.Equal("call-1", resultHistory.RootElement
            .GetProperty("tool_result").GetProperty("call_id").GetString());
    }

    private static LlamaToolCallCatalog CreateCatalog() => LlamaToolCallCatalog.Create(
    [
        new LlmToolDefinition(
            "local__weather",
            "Gets current weather for one city.",
            Schema),
    ]);
}
