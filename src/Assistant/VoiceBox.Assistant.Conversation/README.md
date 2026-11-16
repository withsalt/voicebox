# VoiceBox.Assistant.Conversation

默认会话实现：有限内存历史、LLM 流式输出、上下文扩展、结构化工具循环与后台结果通道。

## 基础文字会话

```csharp
await using IConversationSession conversation = new ConversationBuilder(llm)
    .UseConversationId("chat-1")
    .UseOptions(new LlmConversationOptions
    {
        SystemPrompt = "请用简洁的中文回答。",
        MaxHistoryTurns = 20,
        MaxResponseTokens = 512,
        EnableThinking = false,
    })
    .Build();

await foreach (ConversationEvent conversationEvent in conversation.StreamAsync(
    new ConversationRequest(Guid.NewGuid(), "你好")))
{
    // ConversationThinkingDelta / ConversationTextDelta / ConversationCompleted
}
```

- `LlmConversationOptions` 在 `UseOptions` 时复制并验证；之后修改调用方对象不会影响运行中的会话。
- 历史由会话拥有。完整生成后先提交历史，再输出 `ConversationCompleted`；取消、异常或提前释放会清理本轮暂存消息。
- 同一会话的请求串行执行；不同会话互不影响。
- 借用调用方的 `IStreamLLMService`；会话创建并拥有 provider 的 `ILLMConversation`，释放会话不会释放 LLM 服务。
- 构造完成即可使用，不需要启动步骤。
- `ConversationRequest.Audio` 可携带本轮语音快照；默认实现使用 `Text`，不会自动把音频发送给 LLM 或存入历史。

## 上下文扩展

```csharp
await using IConversationSession conversation = new ConversationBuilder(llm)
    .AddContextProvider(memoryContextProvider)
    .Build();
```

`IConversationContextProvider.GetContextAsync` 每轮在构建模型请求前调用一次，按注册顺序合并返回的背景段。
背景只附加到本轮 system 消息，不会变成永久用户消息；工具续推复用本轮背景。默认检索失败使本轮失败且不提交。

## 工具会话

工具公共契约位于 `VoiceBox.Assistant.Conversation.Abstractions.Tools`。默认会话接收
`IConversationTool` 和 `IConversationToolProvider`：

```csharp
using VoiceBox.Assistant.Conversation.Abstractions.Tools;
using VoiceBox.Assistant.Conversation.Mcp;

// MCP 来源由调用方创建；创建阶段只冻结配置，不连接网络。
IConversationToolProvider files = McpToolProvider.CreateStdio("filesystem", options =>
{
    options.Command = configuredCommand;
    options.AllowedTools.Add("read_file");
});

IConversationSession conversation = new ConversationBuilder(toolCallingLlm)
    .UseOptions(conversationOptions)
    .UseLoggerFactory(loggerFactory) // 可选；未配置时不输出日志
    .UseTools(tools =>
    {
        tools.AddTool(new CurrentWeatherTool()); // IConversationTool
        tools.AddToolProvider(files);
        tools.UseApprovalPolicy(approvalPolicy);
    })
    .Build();

var lifecycle = (IConversationLifecycle)conversation;
await lifecycle.StartAsync();
try
{
    // StreamAsync...
}
finally
{
    await lifecycle.StopAsync();
    await conversation.DisposeAsync();
}
```

- 会话在 Build 阶段要求 `IToolCallingStreamLLMService`；不支持的 Provider 明确失败。
- `Build` 只校验并冻结配置，不建立网络连接；MCP 连接、进程启动和 `tools/list` 发生在 `StartAsync`。
- 成功 Build 后，本地工具、外部工具来源和工具运行时会话拥有；`DisposeAsync` 依次停止后台工作、释放来源与工具。
  同一个工具或来源实例不能注册到多个拥有型会话。
- 工具运行时负责审批、JSON Schema 参数校验、调用顺序、超时、重试和 `UnknownOutcome` 不重放。
- 后台工具结果通过 `IConversationBackgroundOutput` 发布；消息在入队前已经写入会话历史。通道有界（容量 64）并使用
  Wait 背压，`StopAsync` 会取消阻塞写入；下一次 `StartAsync` 创建新通道，不重放上一运行的残留消息。

### 流式输出

工具会话与基础文字会话一样，模型每产生一段可见文字就立即输出，不等整轮生成结束。

- 模型在发起工具调用之前说出的文字（例如“我查一下。”）照常输出，并计入 `ConversationCompleted.Text` 和会话历史。
  工具执行之后的文字另起一行（以 `\n` 分隔），不与之前的文字粘连。
- 这段文字会随同一条 assistant 消息（文字 + 工具调用）回传给模型，模型据此接着往下说。自定义的
  `IToolCallingStreamLLMService` 如果会在同一轮既输出文字又发起调用，需要接受这种消息。内置的 OpenAI 实现已支持；
  LLamaSharp 每轮只输出文字或调用之一，不会收到这种消息。
- 工具自己的 `ExecutionPrompt` 以 `ConversationTextDelta(prompt, EndsTextSegment: true)` 输出；消费者继续拉取上游后才执行工具。
  模型文字不带这个标记。

### 调用的执行控制

- **超时与取消由会话强制执行。** 每次调用的时限是工具自己的 `ConversationToolDescriptor.Timeout`。到期或请求被取消后，
  会话立即停止等待并继续本轮；不响应取消令牌的工具会被留在后台，它迟到的结果被丢弃并记入日志。
  只读工具超时记为 `TimedOut`，有副作用的工具超时记为 `UnknownOutcome`。
- **同一批调用按模型给出的顺序生效。** 审批逐个串行。连续的无副作用调用（`ReadOnly` / `Sensitive`）并行执行，
  并发数不超过 `MaxParallelToolCalls`；有副作用的调用等它前面的调用全部结束后才开始，并在它后面的调用开始前结束。
- **重试只针对只读工具。** 触发条件是超时，或结果标记了 `ConversationToolResult.IsTransient`；工具直接抛出的
  `HttpRequestException`、`TimeoutException`、`SocketException` 和 `IOException`（不含 `FileNotFoundException` 等子类）
  也视为瞬时故障。`MaxToolRetryBudgetMs` 只约束重试：首次调用不受它限制，预算用尽后不再重试。

### 固定播报文字

会话自己生成的文字只有两类：后台执行提示（工具没有配置 `ExecutionPrompt` 时使用），以及后台结果无法由模型总结时的兜底说明。
它们集中在 `ConversationToolOptions.Announcements`，默认中文，`{0}` 替换为工具的 `DisplayName`：

```csharp
tools.UseOptions(new ConversationToolOptions
{
    Announcements = new ConversationToolAnnouncements
    {
        BackgroundStarted = "Working on {0}. I'll get back to you.",
        NameSeparator = " and ",
        BackgroundSucceeded = "{0} is done. ",
        // BackgroundDenied / BackgroundCancelled / BackgroundUnknownOutcome / BackgroundFailed
    },
});
```

### 日志

`UseLoggerFactory` 接入后，会话记录：工具调用的非成功结果与重试、工具抛出的异常（含堆栈）、后台任务故障、
被留在后台的调用何时返回、未完成回合的历史回滚，以及停止和释放阶段的故障。日志包含工具身份、调用 ID、结果分类和耗时；
会话自身不写入对话文字、工具参数或工具返回内容（工具抛出的异常消息除外）。

### 发给模型的 JSON

工具 Schema、调用参数和工具结果中的非 ASCII 文字（例如中文）按原文发送，不转成 `\uXXXX` 转义；`<`、`>`、`&` 等字符仍然转义。
`MaxPersistedToolResultBytes` 按实际 UTF-8 字节数计算。

工具默认值见 `ConversationToolOptions`。不依赖默认会话的实现路径见
[Abstractions](../VoiceBox.Assistant.Conversation.Abstractions/README.md) 与
[MCP 工具来源](../VoiceBox.Assistant.Conversation.Mcp/README.md)；后者只依赖 Abstractions，不要求安装本默认实现。

## 示例

- [ConversationTextDemo](../Demos/ConversationTextDemo)：不创建音频设备的独立文字对话，包含上下文记忆示例。
- [VoiceAssistantDemo](../Demos/VoiceAssistantDemo)：语音助手接入默认工具会话。
- `src/Assistant/Test/VoiceBox.Assistant.Conversation.Tests`：取消、提交竞争、上下文、工具策略、后台通道与 MCP 配置测试。
