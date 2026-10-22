using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.LLamaSharp;

namespace LLamaSharpLLMDemo
{
    /// <summary>
    /// LLamaSharp GGUF LLM 端到端示例。覆盖三种典型用法:
    /// <list type="number">
    ///   <item>单轮文本流式问答 (text → text)</item>
    ///   <item>多轮 chat 上下文累积 (有状态会话 + KV-cache 复用)</item>
    ///   <item>Thinking 模式 (模型推理过程侧显)</item>
    /// </list>
    /// </summary>
    internal static class Program
    {
        // 将 ModelPath 改为你机器上的 GGUF 文件路径即可。
        // 推荐入门: MiniCPM5-2B-Q4_K_M.gguf (~1.45GB, CPU 友好)。
        private const string BASE_MODEL_DIR = @"D:\ML\Audio\voicebox_models\v1";
        private static readonly string ModelPath = Path.Combine(BASE_MODEL_DIR, "llm", "llama", "MiniCPM5-2B-GGUF", "MiniCPM5-2B-Q4_K_M.gguf");
        private const string AssistantName = "LLamaSharp";

        private static async Task Main(string[] args)
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            if (!System.IO.File.Exists(ModelPath))
            {
                Console.WriteLine($"模型文件不存在: {ModelPath}");
                Console.WriteLine("请修改 Program.cs 中的 ModelPath, 指向本机的 GGUF 文件。");
                return;
            }

            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                // Demo 保持聊天界面干净; LLamaSharp 的原生日志由此过滤。
                builder.SetMinimumLevel(LogLevel.Warning);
                builder.AddSimpleConsole(o =>
                {
                    o.SingleLine = true;
                    o.TimestampFormat = "HH:mm:ss.fff ";
                });
            });

            var options = Options.Create(new LLamaSharpLLMOptions
            {
                ModelPath = ModelPath,
                ContextSize = 4096,
                GpuLayerCount = -1,        // 纯 CPU; 有 GPU 时改为 -1 (全部卸载)
                MaxTokens = 4096,
                // 以下采样参数适合对话场景
                DoSample = true,
                Temperature = 0.7f,
                TopK = 40,
                TopP = 0.9f,
                RepetitionPenalty = 1.1f,
                EnableSessionCache = true,
            });

            Console.WriteLine($"加载模型中: {ModelPath} ...");
            var sw = Stopwatch.StartNew();
            using var llm = new LLamaSharpLLMService(
                options, loggerFactory.CreateLogger<LLamaSharpLLMService>());
            sw.Stop();
            Console.WriteLine($"模型已就绪 (加载耗时 {sw.Elapsed.TotalSeconds:F1}s)。");
            Console.WriteLine($"Profile: {llm.GetType().Name}");
            Console.WriteLine();

            string mode = SelectMode(args);
            switch (mode)
            {
                case "1": await RunTextStreaming(llm); break;
                case "2": await RunMultiTurnChat(llm); break;
                case "3": await RunThinkingMode(llm); break;
                default:
                    Console.WriteLine("Unknown mode.");
                    break;
            }
        }

        private static string SelectMode(string[] args)
        {
            if (args.Length > 0 && args[0] is "1" or "2" or "3")
                return args[0];

            Console.WriteLine("请选择模式:");
            Console.WriteLine("  1. 文本流式问答");
            Console.WriteLine("  2. 多轮 chat (有状态会话 + KV-cache 复用)");
            Console.WriteLine("  3. Thinking 模式");
            Console.Write("> ");
            return (Console.ReadLine() ?? "1").Trim();
        }

        // ---------- 模式 1: 单轮文本流式问答 ----------

        private static async Task RunTextStreaming(IStreamLLMService llm)
        {
            Console.WriteLine("=== 文本流式问答 ===");
            Console.WriteLine("输入 :q 退出。");
            Console.WriteLine();

            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line is ":q" or ":quit") break;

                using var cts = new CancellationTokenSource();
                Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

                var sw = Stopwatch.StartNew();
                Console.Write($"\n{AssistantName}> ");
                int chunkCount = 0;

                try
                {
                    await foreach (var chunk in llm.StreamAsync(
                        [ChatMessage.User(line)],
                        new LlmCompleteOptions { MaxTokens = 512 },
                        cts.Token))
                    {
                        if (chunk.IsFinal) continue;
                        if (chunk.Kind == LlmChunkKind.Text)
                        {
                            Console.Write(chunk.Text);
                            chunkCount++;
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    Console.Write("\n(已中断)");
                }
                finally
                {
                    sw.Stop();
                    Console.WriteLine($"\n[{sw.Elapsed.TotalSeconds:F1}s, {chunkCount} chunks]");
                }
            }
        }

        // ---------- 模式 2: 多轮 chat ----------

        private static async Task RunMultiTurnChat(IConversationalLLMService llm)
        {
            Console.WriteLine("=== 多轮 Chat (KV-cache 复用) ===");
            Console.WriteLine("输入 :q 退出, :reset 新开对话。");
            Console.WriteLine();

            var history = new List<ChatMessage>();
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

            using var conversation = llm.CreateConversation();

            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line is ":q" or ":quit") break;
                if (line is ":reset")
                {
                    Console.WriteLine("(对话已重置)");
                    history.Clear();
                    continue;
                }

                history.Add(ChatMessage.User(line));

                var sw = Stopwatch.StartNew();
                Console.Write($"\n{AssistantName}> ");
                var responseTokens = new System.Text.StringBuilder();

                try
                {
                    await foreach (var chunk in conversation.StreamAsync(
                        history,
                        new LlmCompleteOptions { MaxTokens = 1024 },
                        cts.Token))
                    {
                        if (chunk.IsFinal) continue;
                        if (chunk.Kind == LlmChunkKind.Text)
                        {
                            Console.Write(chunk.Text);
                            responseTokens.Append(chunk.Text);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    Console.Write("\n(已中断)");
                    // 保留已生成的 partial 回复进入历史, 下轮可复用已计算的 KV。
                }

                string reply = responseTokens.ToString();
                if (!string.IsNullOrEmpty(reply))
                    history.Add(ChatMessage.Assistant(reply));

                sw.Stop();
                Console.WriteLine($"\n[{sw.Elapsed.TotalSeconds:F1}s]");
            }
        }

        // ---------- 模式 3: Thinking 模式 ----------

        private static async Task RunThinkingMode(IStreamLLMService llm)
        {
            Console.WriteLine("=== Thinking 模式 ===");
            Console.WriteLine("推理过程以灰色显示 (终端支持 ANSI 时), 输入 :q 退出。");
            Console.WriteLine();

            while (true)
            {
                Console.Write("\nYou> ");
                string? line = Console.ReadLine();
                if (line is null) break;
                line = line.Trim();
                if (line.Length == 0) continue;
                if (line is ":q" or ":quit") break;

                using var cts = new CancellationTokenSource();
                Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

                bool inThinking = false;
                var sw = Stopwatch.StartNew();
                Console.Write($"\n{AssistantName}> ");

                try
                {
                    await foreach (var chunk in llm.StreamAsync(
                        [ChatMessage.User(line)],
                        new LlmCompleteOptions { MaxTokens = 1024, EnableThinking = true },
                        cts.Token))
                    {
                        if (chunk.IsFinal) continue;

                        if (chunk.Kind == LlmChunkKind.Thinking)
                        {
                            if (!inThinking)
                            {
                                Console.Write("\x1b[90m[思考] ");  // 灰色
                                inThinking = true;
                            }
                            Console.Write(chunk.Text);
                        }
                        else
                        {
                            if (inThinking)
                            {
                                Console.Write("\x1b[0m\n");  // 恢复颜色 + 换行
                                inThinking = false;
                            }
                            Console.Write(chunk.Text);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    Console.Write("\n(已中断)");
                }
                finally
                {
                    if (inThinking) Console.Write("\x1b[0m");
                    sw.Stop();
                    Console.WriteLine($"\n[{sw.Elapsed.TotalSeconds:F1}s]");
                }
            }
        }
    }
}
