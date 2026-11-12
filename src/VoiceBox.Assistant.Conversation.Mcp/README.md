# VoiceBox.Assistant.Conversation.Mcp

把 MCP Server 注册为工具来源。本项目只依赖
[VoiceBox.Assistant.Conversation.Abstractions](../VoiceBox.Assistant.Conversation.Abstractions/README.md)
与 `ModelContextProtocol.*`，不引用默认 Conversation、Assistant 或 Core。

## 创建与生命周期

```csharp
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Mcp;

McpToolProvider provider = McpToolProvider.CreateStdio("filesystem", options =>
{
    options.Command = "npx";
    options.Arguments.Add("-y");
    options.Arguments.Add("@modelcontextprotocol/server-filesystem");
    options.Arguments.Add(workspaceDirectory);
    options.AllowedTools.Add("read_file");
    options.DefaultRisk = ConversationToolRisk.ReadOnly;
    options.ExecutionPolicy = ConversationToolExecutionPolicy.ForegroundOnly;
});

await provider.StartAsync(cancellationToken);
try
{
    IReadOnlyList<ConversationToolDefinition> tools =
        await provider.GetToolsAsync(cancellationToken);
    // 宿主完成工具选择、审批与参数处理后再调用 InvokeAsync。
    ConversationToolResult result = await provider.InvokeAsync(
        new ConversationToolCall("call-1", tools[0].Id, """{"path":"README.md"}"""),
        cancellationToken);
}
finally
{
    await provider.StopAsync();
    await provider.DisposeAsync();
}
```

- `CreateStdio` / `CreateStreamableHttp` 只校验并冻结配置：不启动进程、不连接网络、不发现工具。
  创建后修改原始 options 不会影响 provider。
- 两个创建方法都可以传入可选的 `ILoggerFactory`，接收 MCP client、transport 与连接关闭故障的诊断日志；未提供时不输出日志。
- 一个 provider 对应一个 MCP Server；多个来源由宿主分别创建并组合。
- `StartAsync` 建立 client、启动 stdio 子进程并执行 `tools/list`；启动失败会清理已创建的连接。
- `StopAsync` 结束本次运行，允许随后再次 `StartAsync`；`DisposeAsync` 负责最终释放。
- 拥有 provider 的宿主负责协调生命周期，并在关闭连接前取消、等待自己的活动调用。

## 配合默认会话

默认会话通过通用 `AddToolProvider` 接入；不依赖 MCP 专用 builder 扩展：

```csharp
using VoiceBox.Assistant.Conversation;
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Mcp;

IConversationToolProvider files = McpToolProvider.CreateStdio("filesystem", options =>
{
    options.Command = "npx";
    options.AllowedTools.Add("read_file");
});

await using IConversationSession session = new ConversationBuilder(llm)
    .UseTools(tools => tools.AddToolProvider(files))
    .Build();
```

成功 Build 后 provider 所有权转移给会话；Build 前由调用方负责释放。会话在 `StartAsync` 时才连接 MCP。

## 职责边界

| 层 | 负责 |
| --- | --- |
| 会话工具运行时（默认 Conversation 或调用方自己的宿主） | 工具聚合、稳定命名、审批、参数 Schema 校验、并发/重试预算、结果关联 |
| `McpToolProvider`（本程序集） | transport、服务发现、`tools/call`、MCP Tasks polling、协议错误与结果大小归一化 |
| MCP Server | 实际工具实现 |

- 工具 identity 为 `(ServerAlias, ToolName)`；provider-safe 暴露名由消费会话生成并检测碰撞。
- MCP 返回值只支持文本和 structured content：文本块进入 `ConversationToolResult.Content`，structured content 进入
  `StructuredContentJson`。仅含不支持内容类型时返回 `UnsupportedCapability`，不静默等待或当成成功。
- 参数/结果大小、连接与请求超时、并发数、allowlist、默认风险等级和执行策略来自 server options。
  `MaxResultBytes` 分别限制文本结果和 structured content 的 UTF-8 字节数，任一超出即返回 `result_too_large`。
- `RequestTimeout` 由 provider 执行，覆盖工具调用等待并发名额、请求执行和 MCP Tasks 轮询。
  调用超时返回 `TimedOut`（标记为 `IsTransient`，可以重试）；已发出请求的 `SideEffecting` / `Destructive` /
  `Privileged` 工具返回 `UnknownOutcome`，调用方应避免重复执行。两种结果的 `ErrorCode` 都是 `timeout`。
  调用方主动取消仍抛出 `OperationCanceledException`。
- 直接调用 `IConversationToolProvider.InvokeAsync` 不会自动执行默认会话的审批、重试或历史逻辑。

## 支持范围

支持 stdio、Streamable HTTP 与 MCP Tasks。resources、prompts、sampling、elicitation、Apps 和
完整 OAuth 流程不在当前范围内。

Provider 能力：`VoiceBox.LLM.OpenAI` 与 `VoiceBox.LLM.LLamaSharp` 实现结构化工具调用；
`VoiceBox.LLM.OnnxGenAI` 不实现 `IToolCallingStreamLLMService`，不能构建默认工具会话。
