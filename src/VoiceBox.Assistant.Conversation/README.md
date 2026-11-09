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
- 工具运行时负责审批、JSON Schema 参数校验、只读并发、副作用串行、重试预算和 `UnknownOutcome` 不重放。
- 工具提示以 `ConversationTextDelta(prompt, EndsTextSegment: true)` 输出；消费者继续拉取上游后才执行工具。
- 后台工具结果通过 `IConversationBackgroundOutput` 发布；消息在入队前已经写入会话历史。通道有界（容量 64）并使用
  Wait 背压，`StopAsync` 会取消阻塞写入；下一次 `StartAsync` 创建新通道，不重放上一运行的残留消息。

工具默认值见 `ConversationToolOptions`。不依赖默认会话的实现路径见
[Abstractions](../VoiceBox.Assistant.Conversation.Abstractions/README.md) 与
[MCP 工具来源](../VoiceBox.Assistant.Conversation.Mcp/README.md)；后者只依赖 Abstractions，不要求安装本默认实现。

## 示例

- [ConversationTextDemo](../Demos/ConversationTextDemo)：不创建音频设备的独立文字对话，包含上下文记忆示例。
- [VoiceAssistantDemo](../Demos/VoiceAssistantDemo)：语音助手接入默认工具会话。
- `src/Tests/VoiceBox.Assistant.Conversation.Tests`：取消、提交竞争、上下文、工具策略、后台通道与 MCP 配置测试。
