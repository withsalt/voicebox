using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OnnxGenAI;

namespace OnnxGenAILLMDemo
{
    /// <summary>
    /// OnnxGenAI LLM 端到端示例。覆盖三种典型用法:
    /// <list type="number">
    ///   <item>单轮文本流式问答 (text → text)</item>
    ///   <item>多轮 chat 上下文累积</item>
    ///   <item>音频转写 (audio → text), 与传统 ASR 不同, 这里模型能直接理解语义并按 prompt 输出。</item>
    /// </list>
    /// </summary>
    internal static class Program
    {
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
        private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "llm", "genai", "Qwen3-1.7B-ONNX-Q4-CPU");
        private const GenAIExecutionProvider ExecutionProvider = GenAIExecutionProvider.Cpu;
        private const string AssistantName = "OnnxGenAI";


        private static async Task Main(string[] args)
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            if (!Directory.Exists(ModelPath))
            {
                Console.WriteLine($"模型目录不存在: {ModelPath}");
                Console.WriteLine("请先按 VoiceBox.LLM.OnnxGenAI/README.md 的指引准备 ORT-GenAI 兼容的模型。");
                return;
            }

            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                // Demo 默认保持聊天界面干净; 服务层的诊断日志避免穿插到流式回答中。
                builder.SetMinimumLevel(LogLevel.Warning);
                builder.AddSimpleConsole(o =>
                {
                    o.SingleLine = true;
                    o.TimestampFormat = "HH:mm:ss.fff ";
                });
            });

            var options = Options.Create(new OnnxGenAILLMOptions
            {
                ModelPath = ModelPath,
                ExecutionProvider = ExecutionProvider,
                MaxTokens = 2048,
                //以下参数配置适用于聊天场景
                DoSample = true,
                TopK = 40,
                Temperature = 0.5f,
                TopP = 0.9f,
                RepetitionPenalty = 1.1f,
            });

            Console.WriteLine($"加载模型中: {ModelPath} (provider={ExecutionProvider}) ...");
            using var llm = new OnnxGenAILLMService(options, loggerFactory.CreateLogger<OnnxGenAILLMService>());
            Console.WriteLine("模型已就绪。");
            Console.WriteLine();

            string mode = SelectMode(args);
            switch (mode)
            {
                case "1": await RunTextStreaming(llm); break;
                case "2": await RunMultiTurnChat(llm); break;
                case "3": await RunAudioTranscription(llm, args); break;
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
            Console.WriteLine("  3. 音频转写 / 理解");
            Console.Write("> ");
            return (Console.ReadLine() ?? "1").Trim();
        }

        private static async Task RunTextStreaming(IStreamLLMService llm)
        {
            Console.WriteLine("=== 文本流式问答 ===");
            Console.WriteLine("输入 :q 退出, :think 显示/隐藏思考过程 (当前: 显示)");
            Console.WriteLine();

            bool showThinking = true;
            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line == ":q" || line == ":quit") break;
                if (line == ":think")
                {
                    showThinking = !showThinking;
                    Console.WriteLine($"(思考显示: {(showThinking ? "开" : "关")})");
                    continue;
                }

                using var cts = new CancellationTokenSource();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var renderer = new ThinkingAwareRenderer(AssistantName, showThinking);
                var callOptions = new LlmCompleteOptions { EnableThinking = showThinking };
                int chars = 0;
                try
                {
                    await foreach (var chunk in llm.StreamAsync(new[] { ChatMessage.User(line) }, callOptions, cts.Token))
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
            Console.WriteLine($"输入 :q 退出, :reset 清空历史, :think 显示/隐藏思考过程 (当前: 显示)");
            Console.WriteLine($"(最多保留最近 {MaxTurns} 轮对话)");

            // 若 LLM 支持有状态对话, 用会话跨轮复用 KV-cache; 否则退化为无状态 StreamAsync。
            using var conversation = (llm as IConversationalLLMService)?.CreateConversation();
            if (conversation is not null)
                Console.WriteLine("(已启用会话 KV-cache 复用)");
            Console.WriteLine();

            string role = "你是一个自然、真实、简洁的日常聊天伙伴。和我对话时，不要使用机器化、模板化的表达，也不要频繁提到“AI”“模型”“程序”等身份信息。\r\n你的说话方式要像一个普通人：\r\n1. 语气自然，像朋友聊天。\r\n2. 回答简洁，不说废话。\r\n3. 不要过度解释，除非我追问。\r\n4. 不要用“作为AI”这类开场。\r\n5. 不要刻意客套，不要假装很懂。\r\n6. 可以有正常的语气词，比如“嗯”“确实”“我懂”“可以”。\r\n7. 如果不确定，就直接说“不太确定”。\r\n8. 根据我的语气调整回应，轻松、自然、有分寸。\r\n你不需要强调自己的身份，只需要像一个真实的聊天对象一样回应我。";

            var systemMessage = ChatMessage.System(role);
            var history = new List<ChatMessage> { systemMessage };
            bool showThinking = true;

            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line == ":q" || line == ":quit") break;
                if (line == ":think")
                {
                    showThinking = !showThinking;
                    Console.WriteLine($"(思考显示: {(showThinking ? "开" : "关")})");
                    continue;
                }
                if (line == ":reset")
                {
                    history.Clear();
                    history.Add(systemMessage);
                    Console.WriteLine("(已清空历史)");
                    continue;
                }

                history.Add(ChatMessage.User(line));

                var renderer = new ThinkingAwareRenderer(AssistantName, showThinking);
                var callOptions = new LlmCompleteOptions { EnableThinking = showThinking };
                var reply = new System.Text.StringBuilder();
                try
                {
                    var stream = conversation is not null
                        ? conversation.StreamAsync(history, callOptions)
                        : llm.StreamAsync(history, callOptions);
                    await foreach (var chunk in stream)
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
                    history.RemoveAt(history.Count - 1);  // 失败的 user 消息从历史中撤回
                    continue;
                }
                renderer.Complete();
                Console.WriteLine();
                history.Add(ChatMessage.Assistant(reply.ToString()));

                TrimHistory(history, systemMessage, MaxTurns);
            }
        }

        /// <summary>
        /// 仅保留最近 <paramref name="maxTurns"/> 轮 user/assistant 对话, system 消息始终置顶。
        /// 一轮 = 一条 user + 一条 assistant; 修剪在每次 assistant 回复完整入栈后执行。
        /// </summary>
        private static void TrimHistory(List<ChatMessage> history, ChatMessage systemMessage, int maxTurns)
        {
            int maxConvMessages = maxTurns * 2;  // 每轮 = user + assistant
            // history[0] 是 system, 后面才是对话; 超出窗口时从最早的对话开始丢。
            int convMessages = history.Count - 1;
            if (convMessages <= maxConvMessages) return;

            int dropCount = convMessages - maxConvMessages;
            history.RemoveRange(1, dropCount);

            // 防御性: 第一条对话必须是 user (避免半轮悬空, 影响 chat template 渲染)。
            if (history.Count > 1 && history[1].Role != ChatRole.User)
                history.RemoveAt(1);
        }

        private static async Task RunAudioTranscription(IStreamLLMService llm, string[] args)
        {
            Console.WriteLine("=== 音频转写 / 理解 ===");

            string? audioPath = args.Length > 1 ? args[1] : null;
            if (string.IsNullOrEmpty(audioPath))
            {
                Console.Write("请输入音频文件路径 (.wav/.mp3): ");
                audioPath = Console.ReadLine()?.Trim();
            }
            if (string.IsNullOrEmpty(audioPath) || !File.Exists(audioPath))
            {
                Console.WriteLine("音频文件不存在。");
                return;
            }

            Console.Write("请输入提示 (留空使用 \"Transcribe this audio.\"): ");
            string? prompt = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(prompt)) prompt = "Transcribe this audio.";

            var messages = new[]
            {
                ChatMessage.User(
                    new TextContent(prompt),
                    new AudioContent(audioPath)),
            };

            Console.WriteLine();
            var renderer = new ThinkingAwareRenderer(AssistantName, showThinking: false);
            try
            {
                await foreach (var chunk in llm.StreamAsync(messages, new LlmCompleteOptions { EnableThinking = false }))
                {
                    renderer.Write(chunk);
                }
                renderer.Complete();
                Console.WriteLine();
            }
            catch (Exception ex)
            {
                renderer.Complete();
                Console.WriteLine($"\n[错误] {ex.Message}");
            }
        }

        /// <summary>
        /// 控制台流式渲染器: 默认隐藏 thinking 块, 需要诊断时可用 :think 打开暗灰色思考输出。
        /// </summary>
        private sealed class ThinkingAwareRenderer
        {
            private readonly string _answerPrefix;
            private readonly bool _showThinking;
            private bool _thinkingOpen;
            private bool _answerStarted;

            public ThinkingAwareRenderer(string assistantName, bool showThinking)
            {
                _answerPrefix = assistantName + "> ";
                _showThinking = showThinking;
            }

            public void Write(LlmTextChunk chunk)
            {
                if (string.IsNullOrEmpty(chunk.Text)) return;

                if (chunk.Kind == LlmChunkKind.Thinking)
                {
                    if (!_showThinking) return;
                    if (!_thinkingOpen)
                    {
                        _thinkingOpen = true;
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.Write("[思考] ");
                    }
                    Console.Write(chunk.Text);
                    return;
                }

                CloseThinking();
                if (!_answerStarted)
                {
                    _answerStarted = true;
                    Console.Write(_answerPrefix);
                }
                Console.Write(chunk.Text);
            }

            public void Complete() => CloseThinking();

            private void CloseThinking()
            {
                if (!_thinkingOpen) return;
                _thinkingOpen = false;
                Console.ResetColor();
                Console.WriteLine();
            }
        }
    }
}
