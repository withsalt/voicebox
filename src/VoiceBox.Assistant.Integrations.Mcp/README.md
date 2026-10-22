# VoiceBox.Assistant MCP / Tools

`VoiceBox.Assistant.Integrations.Mcp` 为 `VoiceBox.Assistant` 提供进程内工具、MCP stdio、
MCP Streamable HTTP 和 MCP Tasks 支持。只有显式调用 `UseMcpTools` 才会安装工具能力；
`Build()` 不会连接外部服务，连接和 `tools/list` 在 `VoiceAssistant.StartAsync()` 完成。

## 注册

```csharp
using VoiceBox.Assistant.Api.Tools;
using VoiceBox.Assistant.Integrations.Mcp;

VoiceAssistant assistant = VoiceAssistant.CreateBuilder()
    .UseAudio(audio)
    .UseTurnBoundary(vad)
    .UseSpeechRecognition(asr)
    .UseResponseGeneration(openAiLlm)
    .UseMcpTools(mcp =>
    {
        mcp.AddTool(new WeatherTool());

        mcp.AddStdioServer("filesystem", options =>
        {
            options.Command = "npx";
            options.Arguments.Add("-y");
            options.Arguments.Add("@modelcontextprotocol/server-filesystem");
            options.Arguments.Add(@"D:\data");
            options.DisplayName = "文件处理";
            options.AllowedTools.Add("read_file");
            options.DefaultRisk = AssistantToolRisk.Sensitive;
        });

        mcp.AddStreamableHttpServer("crm", options =>
        {
            options.Endpoint = new Uri("https://crm.example.com/mcp");
            options.Headers["Authorization"] = "Bearer ...";
            options.DisplayName = "客户信息查询";
            options.AllowedTools.Add("find_customer");
            options.DefaultRisk = AssistantToolRisk.ReadOnly;
            options.UseMcpTasks = true;
        });

        mcp.UseApprovalPolicy(new ProductToolApprovalPolicy());
    })
    .UseSpeechSynthesis(tts)
    .Build();
```

首期只接受实现 `IToolCallingLLMService` 的 LLM；当前内置实现只有
`VoiceBox.LLM.OpenAI.OpenAILLMService`。LLamaSharp 或 OnnxGenAI 与 `UseMcpTools` 组合时，
`Build()` 会明确失败，不会解析普通文本模拟工具调用。

## 进程内工具

```csharp
public sealed class WeatherTool : IAssistantTool
{
    public AssistantToolDescriptor Descriptor { get; } = new(
        name: "weather",
        displayName: "查询天气",
        description: "查询指定城市的天气",
        inputSchemaJson: """
            {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
            """,
        risk: AssistantToolRisk.ReadOnly,
        executionPolicy: AssistantToolExecutionPolicy.ModelChoice,
        executionPrompt: "正在查询，请稍后");

    public ValueTask<AssistantToolExecutionResult> ExecuteAsync(
        AssistantToolInvocation invocation,
        CancellationToken cancellationToken)
    {
        string city = invocation.Arguments.GetProperty("city").GetString()!;
        return ValueTask.FromResult(AssistantToolExecutionResult.Success(
            $"{city}晴，25度",
            $$"""{"city":"{{city}}","temperature":25}"""));
    }

    public void Dispose() { } // 无自有资源时可留空。
}
```

通过 `AddTool` 注册的本地工具随 `VoiceAssistant.Dispose()` 一起释放。助手先取消并等待
前台和后台调用结束，再逐个释放工具；未启动或启动失败时也会释放。`StopAsync()` 仅停止
运行，不释放本地工具。工具的 `Dispose()` 应支持重复调用，单个工具释放失败不会阻止
其他工具的清理。

`DisplayName` 是宿主提供的受信任名称，会用于后台确认和结果播报；MCP Server 自报的
title 不会进入这段语音。未提供审批策略时仅自动允许 `ReadOnly` 工具，其余风险等级默认拒绝。

## 前后台语义

- `ForegroundOnly`：工具完成并把结果返回模型后，才生成本轮可见回复。
- `BackgroundOnly`：本地工具立即播放其 `ExecutionPrompt`；标准 MCP 工具保持原有 Host 确认文案。
- `ModelChoice`：模型通过 provider 原生 schema 中的保留字段选择；确定为后台时按上述规则提示。
- 本地工具使用 `BackgroundOnly` 或 `ModelChoice` 时必须提供非空白 `ExecutionPrompt`。
- 后台结果生成后进入同一个 FIFO TTS 队列；已有语音播放时等待排空，否则立即播放。
- 有副作用的工具超时记为 `UnknownOutcome`，不会自动重试。

stdio 子进程默认不继承全部父进程环境，只继承 SDK 的安全基础变量和显式配置项。

## 执行策略

工具行为由 `VoiceAssistantOptions.Tools` 控制，全部有可用默认值：

| 选项 | 默认 | 说明 |
| --- | --- | --- |
| `ToolTranscriptRetentionTurns` | 3 | 最近几轮在模型 prompt 中保留 tool call / result。更早的轮次退化为纯文本，避免长会话撑爆 context |
| `MaxPersistedToolResultBytes` | 8 KB | 单条 result 落库上限，超限截断并标注 `__truncated` |
| `MaxParallelToolCalls` | 4 | 同批调用的并发上限，取 1 即全部串行 |
| `MaxToolRetryAttempts` | 2 | 瞬时失败重试次数，仅 `ReadOnly` 工具适用 |
| `ToolRetryBaseDelayMs` | 200 | 首次重试退避，之后每次 ×3 |
| `MaxToolRetryBudgetMs` | 60000 | 含全部重试的单调用总耗时上限 |
| `ValidateToolArguments` | true | 在宿主侧按 input schema 校验模型参数 |

要点：

- **跨轮工具记忆**：工具调用与结果会进入模型历史，下一轮追问（「那穿短袖行吗」）能看到上一轮的
  结构化结果，而不只是助手的口播摘要。tool result 不进入用户可见历史。
- **并发**：同批调用中 `ReadOnly` / `Sensitive` 工具并发执行，总延迟接近最慢的那一个而非全部之和；
  `SideEffecting` 及以上风险始终按调用序串行，使外部副作用顺序确定。**授权始终串行**，语音确认
  不会同时弹两个问题。
- **参数校验**：非法参数在任何工具代码运行前被拒，模型收到 `InputRequired` + 违规路径和原因，
  可自行修正重试；工具实现不必再防御模型的类型错误。
- **重试**：只有 `ReadOnly` 工具的网络类瞬时失败会重试。工具基于请求本身返回的业务错误不重试。

```csharp
var options = new VoiceAssistantOptions();
options.Tools.MaxParallelToolCalls = 1;          // 需要完全串行时
options.Tools.ToolTranscriptRetentionTurns = 5;  // 追问工具结果很频繁时放宽
```
