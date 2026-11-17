using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Conversation;
using VoiceBox.Assistant.Conversation.Abstractions;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OpenAI;

namespace ConversationTextDemo;

/// <summary>
/// 独立文字会话示例: 不创建音频设备、不启动 Assistant, 直接消费
/// <see cref="IConversationSession.StreamAsync"/>。设置 OPENAI_API_KEY 时接入 OpenAI 兼容端点,
/// 并要求设置 OPENAI_BASE_URL 和 OPENAI_MODEL；未设置 OPENAI_API_KEY 时使用进程内的演示模型。
/// </summary>
internal static class Program
{
    private static async Task Main()
    {
        if (OperatingSystem.IsWindows())
        {
            try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
            try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        }

        using var loggerFactory = LoggerFactory.Create(builder => builder.SetMinimumLevel(LogLevel.Warning));
        using IStreamLLMService llm = CreateLlm(loggerFactory);
        var memory = new NameMemoryContextProvider();

        await using IConversationSession conversation = new ConversationBuilder(llm)
            .UseConversationId("text-demo")
            .UseOptions(new LlmConversationOptions
            {
                SystemPrompt = "你是一个简洁的中文聊天伙伴。",
                MaxHistoryTurns = 20,
            })
            .AddContextProvider(memory)
            .Build();

        Console.WriteLine($"会话: {conversation.ConversationId}");
        Console.WriteLine("输入文字开始对话; 输入 exit 结束。");
        while (true)
        {
            Console.Write("> ");
            string? line = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(line) || string.Equals(line.Trim(), "exit", StringComparison.OrdinalIgnoreCase))
            {
                break;
            }

            var request = new ConversationRequest(Guid.NewGuid(), line.Trim());
            string visible = string.Empty;
            bool thinkingOpen = false;
            try
            {
                await foreach (ConversationEvent conversationEvent in conversation.StreamAsync(request))
                {
                    switch (conversationEvent)
                    {
                        case ConversationThinkingDelta thinking:
                            if (!thinkingOpen)
                            {
                                thinkingOpen = true;
                                Console.ForegroundColor = ConsoleColor.DarkGray;
                                Console.Write("[思考] ");
                            }
                            Console.Write(thinking.Text);
                            break;
                        case ConversationTextDelta text:
                            CloseThinking(ref thinkingOpen);
                            Console.Write(text.Text);
                            break;
                        case ConversationCompleted completed:
                            visible = completed.Text;
                            break;
                    }
                }
            }
            catch (OperationCanceledException)
            {
                CloseThinking(ref thinkingOpen);
                Console.WriteLine("[已取消]");
                continue;
            }
            finally
            {
                // 思考中途取消/异常时也要恢复颜色, 否则后续输出会一直保持暗灰。
                if (thinkingOpen)
                {
                    thinkingOpen = false;
                    Console.ResetColor();
                }
            }

            Console.WriteLine();
            if (visible.Length == 0)
                Console.WriteLine("[空回复]");
        }
    }

    /// <summary>
    /// 结束"思考"分段: 恢复默认前景色并换行, 让随后的回答从新的一行开始。
    /// 没有思考输出时是空操作, 不会多插入空行。
    /// </summary>
    private static void CloseThinking(ref bool thinkingOpen)
    {
        if (!thinkingOpen)
            return;

        thinkingOpen = false;
        Console.ResetColor();
        Console.WriteLine();
    }

    private static IStreamLLMService CreateLlm(ILoggerFactory loggerFactory)
    {
        string? apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            Console.WriteLine("[演示模型] 未设置 OPENAI_API_KEY, 使用进程内回声模型。");
            return new EchoLlmService();
        }

        string baseUrl = GetRequiredEnvironmentVariable("OPENAI_BASE_URL");
        string model = GetRequiredEnvironmentVariable("OPENAI_MODEL");
        Console.WriteLine($"[OpenAI] {baseUrl} (model={model})");
        return new OpenAILLMService(
            Options.Create(new OpenAILLMOptions
            {
                ApiKey = apiKey,
                Endpoint = new Uri(baseUrl),
                Model = model,
                MaxTokens = 1024,
                DoSample = true,
                Temperature = 0.7f,
                TopP = 0.9f,
            }),
            loggerFactory.CreateLogger<OpenAILLMService>());
    }

    private static string GetRequiredEnvironmentVariable(string name)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrWhiteSpace(value)
            ? value
            : throw new InvalidOperationException($"Missing required environment variable: {name}.");
    }

    /// <summary>
    /// 从已提交历史中提取“我叫 X”并作为背景注入下一轮。背景不会变成永久用户消息。
    /// </summary>
    private sealed class NameMemoryContextProvider : IConversationContextProvider
    {
        private string? _name;

        public ValueTask<string> GetContextAsync(ConversationContext context, CancellationToken cancellationToken = default)
        {
            foreach (ChatMessage message in context.CommittedHistory)
            {
                if (message.Role != ChatRole.User)
                    continue;
                string text = string.Concat(message.Content.OfType<TextContent>().Select(c => c.Text));
                int marker = text.IndexOf("我叫", StringComparison.Ordinal);
                if (marker >= 0 && text.Length > marker + 2)
                    _name = text[(marker + 2)..].Trim().TrimEnd('。', '！', '!', '.', '，', ',');
            }

            return ValueTask.FromResult(_name is null ? string.Empty : $"用户的名字是 {_name}。请在合适时自然地称呼对方。");
        }
    }

    /// <summary>无外部依赖的演示模型: 复述输入并说明本轮看到了哪些背景。</summary>
    private sealed class EchoLlmService : IStreamLLMService
    {
        public Task<LlmCompletionResult> CompleteAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            CancellationToken ct = default)
            => Task.FromResult(new LlmCompletionResult { Text = "演示回答。" });

        public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
            IReadOnlyList<ChatMessage> messages,
            LlmCompleteOptions? options = null,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.Yield();
            string system = messages.Count > 0
                ? string.Concat(messages[0].Content.OfType<TextContent>().Select(c => c.Text))
                : string.Empty;
            string lastUser = messages.LastOrDefault(message => message.Role == ChatRole.User) is { } user
                ? string.Concat(user.Content.OfType<TextContent>().Select(c => c.Text))
                : string.Empty;
            string memory = system.Contains("用户的名字是", StringComparison.Ordinal) ? "（我记得你的名字）" : string.Empty;
            string response = $"演示模型{memory}收到: {lastUser}。设置 OPENAI_API_KEY、OPENAI_BASE_URL 和 OPENAI_MODEL 可接入真实模型。";

            foreach (string piece in Split(response, 6))
            {
                ct.ThrowIfCancellationRequested();
                yield return new LlmTextChunk { Text = piece, Kind = LlmChunkKind.Text };
            }
            yield return new LlmTextChunk { Text = string.Empty, Kind = LlmChunkKind.Text, IsFinal = true };
        }

        private static IEnumerable<string> Split(string value, int size)
        {
            for (int offset = 0; offset < value.Length; offset += size)
                yield return value.Substring(offset, Math.Min(size, value.Length - offset));
        }

        public void Dispose()
        {
        }
    }
}
