using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OpenAI;

namespace OpenAILLMDemo
{
    /// <summary>
    /// OpenAI 协议 LLM 端到端示例 (以火山方舟 Ark 上的 DeepSeek 为例)。覆盖三种典型用法:
    /// <list type="number">
    ///   <item>单轮文本流式问答 (text → text, 实时打字机效果)</item>
    ///   <item>多轮 chat 上下文累积</item>
    ///   <item>一次性补全 (CompleteAsync), 展示耗时 / token 粗略指标</item>
    /// </list>
    ///
    /// 凭据通过环境变量提供:
    /// <c>OPENAI_API_KEY</c> / <c>OPENAI_BASE_URL</c> / <c>OPENAI_MODEL</c>。
    /// </summary>
    internal static class Program
    {
        // 默认接入火山方舟 (Ark) 的 OpenAI 兼容端点。
        private const string DefaultBaseUrl = "https://api.deepseek.com";
        private const string DefaultModel = "deepseek-flash";

        private static async Task Main(string[] args)
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.SetMinimumLevel(LogLevel.Information);
                builder.AddSimpleConsole(o =>
                {
                    o.SingleLine = true;
                    o.TimestampFormat = "HH:mm:ss.fff ";
                });
            });

            string apiKey = GetRequiredEnvironmentVariable("OPENAI_API_KEY");
            string baseUrl = Environment.GetEnvironmentVariable("OPENAI_BASE_URL") ?? DefaultBaseUrl;
            string model = Environment.GetEnvironmentVariable("OPENAI_MODEL") ?? DefaultModel;

            var options = Options.Create(new OpenAILLMOptions
            {
                ApiKey = apiKey,
                Endpoint = new Uri(baseUrl),
                Model = model,
                MaxTokens = 2048,
                DoSample = true,
                Temperature = 0.7f,
                TopP = 0.9f,
            });

            Console.WriteLine($"连接 OpenAI 兼容端点: {baseUrl} (model={model}) ...");
            using var llm = new OpenAILLMService(options, loggerFactory.CreateLogger<OpenAILLMService>());
            Console.WriteLine("就绪。");
            Console.WriteLine();

            string mode = SelectMode(args);
            switch (mode)
            {
                case "1": await RunTextStreaming(llm); break;
                case "2": await RunMultiTurnChat(llm); break;
                case "3": await RunOneShotComplete(llm); break;
                default:
                    Console.WriteLine("Unknown mode.");
                    break;
            }
        }

        private static string SelectMode(string[] args)
        {
            if (args.Length > 0 && (args[0] == "1" || args[0] == "2" || args[0] == "3"))
                return args[0];

            Console.WriteLine("请选择模式:");
            Console.WriteLine("  1. 文本流式问答");
            Console.WriteLine("  2. 多轮 chat");
            Console.WriteLine("  3. 一次性补全 (CompleteAsync)");
            Console.Write("> ");
            return (Console.ReadLine() ?? "1").Trim();
        }

        private static string GetRequiredEnvironmentVariable(string name)
            => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value
                ? value
                : throw new InvalidOperationException($"Missing required environment variable: {name}.");

        private static async Task RunTextStreaming(IStreamLLMService llm)
        {
            Console.WriteLine("=== 文本流式问答 ===");
            Console.WriteLine("输入 :q 退出");
            Console.WriteLine();

            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line == ":q" || line == ":quit") break;

                using var cts = new CancellationTokenSource();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var renderer = new ThinkingAwareRenderer();
                int chars = 0;
                try
                {
                    await foreach (var chunk in llm.StreamAsync(new[] { ChatMessage.User(line) }, ct: cts.Token))
                    {
                        renderer.Write(chunk);
                        if (chunk.Kind == LlmChunkKind.Text)
                            chars += chunk.Text.Length;
                    }
                }
                catch (Exception ex)
                {
                    renderer.Complete();
                    Console.WriteLine($"\n[错误] {ex.Message}");
                    continue;
                }
                renderer.Complete();
                sw.Stop();
                Console.WriteLine($"\n[完成] {chars} 字, {sw.Elapsed.TotalSeconds:F2}s");
            }
        }

        private static async Task RunMultiTurnChat(IStreamLLMService llm)
        {
            const int MaxTurns = 20;

            Console.WriteLine("=== 多轮 chat ===");
            Console.WriteLine($"输入 :q 退出, :reset 清空历史 (最多保留最近 {MaxTurns} 轮对话)");
            Console.WriteLine();

            string role = "你是一个自然、真实、简洁的日常聊天伙伴。和我对话时不要使用机器化、模板化的表达，"
                + "也不要频繁提到\"AI\"\"模型\"\"程序\"等身份信息。回答简洁，不说废话，语气像朋友聊天。";

            var systemMessage = ChatMessage.System(role);
            var history = new List<ChatMessage> { systemMessage };

            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line == ":q" || line == ":quit") break;
                if (line == ":reset")
                {
                    history.Clear();
                    history.Add(systemMessage);
                    Console.WriteLine("(已清空历史)");
                    continue;
                }

                history.Add(ChatMessage.User(line));

                var renderer = new ThinkingAwareRenderer();
                var reply = new System.Text.StringBuilder();
                try
                {
                    await foreach (var chunk in llm.StreamAsync(history))
                    {
                        renderer.Write(chunk);
                        // 只把正式回答累积进历史; 思考过程不回传给模型, 避免污染下一轮上下文。
                        if (chunk.Kind == LlmChunkKind.Text)
                            reply.Append(chunk.Text);
                    }
                }
                catch (Exception ex)
                {
                    renderer.Complete();
                    Console.WriteLine($"\n[错误] {ex.Message}");
                    history.RemoveAt(history.Count - 1);
                    continue;
                }
                renderer.Complete();
                Console.WriteLine();
                history.Add(ChatMessage.Assistant(reply.ToString()));

                TrimHistory(history, systemMessage, MaxTurns);
            }
        }

        private static void TrimHistory(List<ChatMessage> history, ChatMessage systemMessage, int maxTurns)
        {
            int maxConvMessages = maxTurns * 2;
            int convMessages = history.Count - 1;
            if (convMessages <= maxConvMessages) return;

            int dropCount = convMessages - maxConvMessages;
            history.RemoveRange(1, dropCount);

            if (history.Count > 1 && history[1].Role != ChatRole.User)
                history.RemoveAt(1);
        }

        private static async Task RunOneShotComplete(ILLMService llm)
        {
            Console.WriteLine("=== 一次性补全 (CompleteAsync) ===");
            Console.WriteLine("输入 :q 退出");
            Console.WriteLine();

            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line == ":q" || line == ":quit") break;

                try
                {
                    var result = await llm.CompleteAsync(new[] { ChatMessage.User(line) });
                    if (!string.IsNullOrEmpty(result.Thinking))
                    {
                        // 思考过程用暗灰色, 与正式回答区分开。
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"[AI Thinking] {result.Thinking}");
                        Console.ResetColor();
                    }
                    Console.WriteLine($"AI> {result.Text}");
                    Console.WriteLine($"[完成] ~{result.TotalTokens} 字符, {result.ElapsedSeconds:F2}s");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[错误] {ex.Message}");
                }
            }
        }

        /// <summary>
        /// 流式渲染器: 把 <see cref="LlmChunkKind.Thinking"/> 与 <see cref="LlmChunkKind.Text"/>
        /// 两类增量分段显示到控制台。
        ///
        /// 思考过程用暗灰色实时滚动 (前缀 "[AI Thinking] "); 收到首个正式回答块时, 自动收束思考段、
        /// 换行并切回正常色显示回答 (前缀 "AI> ")。对没有思考过程的普通模型则直接显示回答, 不留空段。
        ///
        /// 这是把"思考过程"展示出来的推荐范式: UI 只需按 <see cref="LlmTextChunk.Kind"/> 路由,
        /// 服务层不必关心呈现方式。换到 GUI/Web 时, 把"暗灰滚动"换成"可折叠的灰色面板"即可。
        /// </summary>
        private sealed class ThinkingAwareRenderer
        {
            private bool _thinkingOpen;
            private bool _answerStarted;

            public void Write(LlmTextChunk chunk)
            {
                if (string.IsNullOrEmpty(chunk.Text)) return; // 跳过末尾的空 IsFinal 标记块

                if (chunk.Kind == LlmChunkKind.Thinking)
                {
                    if (!_thinkingOpen)
                    {
                        _thinkingOpen = true;
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.Write("[AI Thinking] ");
                    }
                    Console.Write(chunk.Text);
                }
                else
                {
                    CloseThinking();
                    if (!_answerStarted)
                    {
                        _answerStarted = true;
                        Console.Write("AI> ");
                    }
                    Console.Write(chunk.Text);
                }
            }

            /// <summary>确保思考段被收束、控制台颜色被还原 —— 即使流以异常或纯思考结束。可重复调用。</summary>
            public void Complete() => CloseThinking();

            private void CloseThinking()
            {
                if (!_thinkingOpen) return;
                _thinkingOpen = false;
                Console.ResetColor();
                Console.WriteLine(); // 思考段与回答之间换行分隔
            }
        }
    }
}
