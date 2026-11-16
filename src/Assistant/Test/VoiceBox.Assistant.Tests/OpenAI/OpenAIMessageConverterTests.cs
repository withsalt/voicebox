using OpenAI.Chat;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OpenAI.Internal;
using Xunit;
using ChatMessage = VoiceBox.Core.Services.LLMService.Models.ChatMessage;

namespace VoiceBox.Assistant.Tests;

public sealed class OpenAIMessageConverterTests
{
    private static ToolCallContent WeatherCall() => new("c1", "local__weather", """{"city":"北京"}""");

    [Fact]
    public void TextSaidBeforeToolCallsIsSentTogetherWithThoseCalls()
    {
        var message = new ChatMessage(ChatRole.Assistant, [new TextContent("我查一下。"), WeatherCall()]);

        var converted = Assert.IsType<AssistantChatMessage>(Assert.Single(OpenAIMessageConverter.Convert([message])));

        Assert.Equal("我查一下。", Assert.Single(converted.Content).Text);
        ChatToolCall call = Assert.Single(converted.ToolCalls);
        Assert.Equal("c1", call.Id);
        Assert.Equal("local__weather", call.FunctionName);
        Assert.Equal("""{"city":"北京"}""", call.FunctionArguments.ToString());
    }

    [Fact]
    public void ToolCallsWithoutTextCarryNoContent()
    {
        var converted = Assert.IsType<AssistantChatMessage>(
            Assert.Single(OpenAIMessageConverter.Convert([ChatMessage.AssistantToolCalls(WeatherCall())])));

        Assert.Empty(converted.Content);
        Assert.Equal("c1", Assert.Single(converted.ToolCalls).Id);
    }

    [Fact]
    public void AToolCallOutsideAnAssistantTextMessageIsRejected()
    {
        var withImage = new ChatMessage(ChatRole.Assistant, [WeatherCall(), new ImageContent(new byte[] { 1 })]);
        var fromUser = new ChatMessage(ChatRole.User, [new TextContent("你好"), WeatherCall()]);

        Assert.Throws<NotSupportedException>(() => OpenAIMessageConverter.Convert([withImage]));
        Assert.Throws<NotSupportedException>(() => OpenAIMessageConverter.Convert([fromUser]));
    }
}
