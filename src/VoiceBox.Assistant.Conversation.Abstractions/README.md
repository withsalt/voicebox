# VoiceBox.Assistant.Conversation.Abstractions

业务会话的公开契约，仅依赖 BCL。调用方实现 `IConversationSession` 即可把自己的 Agent、存储、记忆或业务流程接入
`VoiceBox.Assistant`，不需要引用默认 LLM 会话、Core 的 ONNX 依赖或任何 Provider SDK。

## 主接口

```csharp
public interface IConversationSession : IAsyncDisposable
{
    string ConversationId { get; }
    IAsyncEnumerable<ConversationEvent> StreamAsync(
        ConversationRequest request,
        CancellationToken cancellationToken = default);
}
```

- `ConversationRequest` 携带本轮文字、调用方生成的 `RequestId` 和可选的 `Audio`；不包含 prompt、历史或模型参数。
- 一次正常请求输出零个或多个 `ConversationTextDelta` / `ConversationThinkingDelta`，随后恰好一个
  `ConversationCompleted`，之后结束。
- `ConversationCompleted.Text` 是全部可见文字增量的顺序拼接；空回复也输出 `Completed("")`。
- 故障通过异步枚举异常传播，取消使用 `OperationCanceledException`。
- 返回流按单次消费设计；提前结束枚举或释放时必须清理未提交状态。

一个只依赖本程序集的业务会话示例：

```csharp
internal sealed class OrderSession : IConversationSession
{
    public string ConversationId => "order-42";

    public async IAsyncEnumerable<ConversationEvent> StreamAsync(
        ConversationRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.Yield();
        cancellationToken.ThrowIfCancellationRequested();
        string answer = await LookupOrderAsync(request.Text, cancellationToken);
        yield return new ConversationTextDelta(answer);
        yield return new ConversationCompleted(answer);
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
```

## 请求音频

`ConversationRequest.Audio` 是与本轮 `Text` 对应的用户语音快照。纯文字输入或没有采样时为 `null`。
音频类型 `ConversationAudio` 只依赖 BCL：

```csharp
var audio = new ConversationAudio(samples, sampleRate: 16_000, channels: 1);
var request = new ConversationRequest(Guid.NewGuid(), "今天的天气如何", audio);

// 自定义会话按需读取；Samples 是 ReadOnlyMemory<float>。
ReadOnlyMemory<float> pcm = request.Audio!.Samples;
```

- `Samples` 为 Float32 PCM，多声道按帧交错排列，不包含 WAV 文件头；`SampleRate`、`Channels` 描述格式，
  `Duration` 根据实际采样帧数计算。
- 构造时复制采样数据；原缓冲区可以复用。快照不需要释放，可以在请求完成、Assistant 停止后继续持有。
- Assistant 传入本轮实际送入 ASR 的音频，包含 ASR 收到的预录音和后续采样，保持顺序一致。
  它可能经过音频前端处理，不表示未经处理的麦克风原始录音，也不提供逐字时间对齐。
- 音频按活动轮次暂存，交给请求后移交给会话。Assistant 不保存历轮音频；会话自行决定是否存储、编码或提交给模型。
- 默认文本会话使用 `Text` 生成模型请求；自定义会话可以直接使用音频进行业务处理。

## 可选能力

- `IConversationLifecycle`：需要建立连接或运行后台工作域的会话实现 `StartAsync` / `StopAsync`。
- `IConversationBackgroundOutput`：请求结束之后仍会产出结果（例如后台工具）的会话实现
  `ReadBackgroundMessagesAsync`。消息携带原始业务会话与请求身份。

不需要这些能力的会话不必声明空实现。默认会话实现见
[VoiceBox.Assistant.Conversation](../VoiceBox.Assistant.Conversation/README.md)。

## 工具契约

工具公共类型位于 `VoiceBox.Assistant.Conversation.Abstractions.Tools`，仅依赖 BCL：

| 类型 | 用途 |
| --- | --- |
| `IConversationTool` | 当前进程承载的具体工具；实现 `Descriptor` 与 `ExecuteAsync` |
| `IConversationToolProvider` | 可启动、发现目录、按身份调用、停止的工具来源 |
| `IConversationToolProviderMetadata` | 可选的连接前静态别名元数据 |
| `IConversationToolApprovalPolicy` | 宿主审批策略 |
| `ConversationToolDescriptor` / `ConversationToolDefinition` | 受信任元数据、Schema 与目录项 |
| `ConversationToolCall` / `ConversationToolResult` | 工具来源的输入与规范化结果 |
| `ConversationToolInvocation` / `ConversationToolExecutionResult` | 本地工具的输入与结果 |
| `ConversationToolId`、风险/执行策略/结果分类枚举 | 公共协作协议 |

第三方程序集可以独立构造调用输入并调用本地工具：

```csharp
using VoiceBox.Assistant.Conversation.Abstractions.Tools;

using JsonDocument document = JsonDocument.Parse("""{"city":"北京"}""");
var invocation = new ConversationToolInvocation(
    "call-1",
    document.RootElement,
    ConversationToolExecutionPreference.Foreground);
// document 释放后 invocation.Arguments 仍然有效。
ConversationToolExecutionResult result = await tool.ExecuteAsync(invocation, cancellationToken);
```

`ConversationToolInvocation` 在构造时验证调用 ID、枚举值和 JSON 对象参数，并复制 `JsonElement`。
Abstractions 不执行 JSON Schema 求值、不调用模型或工具，也不隐含工具/来源的所有权转移。
审批、Schema 校验、并发/重试和历史关联由消费这些契约的会话或宿主决定。

MCP 来源见 [VoiceBox.Assistant.Conversation.Mcp](../VoiceBox.Assistant.Conversation.Mcp/README.md)，
它只依赖本程序集与 MCP SDK。
