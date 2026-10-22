# VoiceBox.LLM.OpenAI

基于官方 [OpenAI .NET SDK](https://github.com/openai/openai-dotnet) 的远程 LLM 服务实现，适配任意
**OpenAI 协议兼容**端点：官方 OpenAI、Azure OpenAI，以及国内的火山方舟 (Ark)、DeepSeek、Moonshot、
阿里百炼等。实现 `VoiceBox.Core` 的 `ILLMService` / `IStreamLLMService` 契约，与 `VoiceBox.LLM.OnnxGenAI`
(本地 ONNX Runtime GenAI) 可无缝替换。

## 设计要点

- **复用底层连接**：`ChatClient` 及其 HTTP 管线在构造期一次性建好，进程内复用，避免每次请求重建连接。
- **客户端并发反压**：用 `SemaphoreSlim` 限制并发上限，降低触发上游 429 限流的概率。
- **流式解耦**：`StreamAsync` 用 `Channel<T>` 解耦生产/消费，`CompleteAsync` 内部消费同一条流，
  风格与 OnnxGenAI 实现一致。
- **推理内容分流**：推理模型 (如 deepseek-reasoner / deepseek-v4) 返回的 `reasoning_content` 增量被
  路由为 `LlmChunkKind.Thinking`，不混入正常文本。对不返回该字段的普通模型是无害的空操作。

## 快速开始

### 直接构造

```csharp
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OpenAI;

var options = Options.Create(new OpenAILLMOptions
{
    ApiKey = "<your-api-key>",
    Endpoint = new Uri("https://ark.cn-beijing.volces.com/api/v3"),
    Model = "deepseek-v4-flash-260425",
    MaxTokens = 4096,
    Temperature = 0.7f,
    SurfaceReasoning = false,
});

using var llm = new OpenAILLMService(options);

await foreach (var chunk in llm.StreamAsync(new[] { ChatMessage.User("用一句话介绍你自己") }))
{
    if (chunk.Kind == LlmChunkKind.Text)
        Console.Write(chunk.Text);
}
```

### 依赖注入

```csharp
services.AddOpenAILLMService(o =>
{
    o.ApiKey = "<your-api-key>";
    o.Endpoint = new Uri("https://ark.cn-beijing.volces.com/api/v3");
    o.Model = "deepseek-v4-flash-260425";
    o.SurfaceReasoning = false;
});
```

或注册到 `AudioServiceManager`：

```csharp
manager.AddOpenAILLMService(o =>
{
    o.ApiKey = "<your-api-key>";
    o.Endpoint = new Uri("https://ark.cn-beijing.volces.com/api/v3");
    o.Model = "deepseek-v4-flash-260425";
});
```

## 协议差异说明

- `Endpoint` 是官方 `base_url`，**不要**带 `/chat/completions` 后缀，SDK 会自动追加。
- OpenAI 协议没有 `top_k` / `repetition_penalty` 字段，因此 `LlmCompleteOptions.TopK` /
  `RepetitionPenalty` 被有意忽略；抑制重复请用 `FrequencyPenalty` / `PresencePenalty`。
- `SurfaceReasoning = false` 只是隐藏/不路由服务端返回的 `reasoning_content`，不会关闭模型在服务端的
  reasoning。
- 火山方舟等 OpenAI-compatible 端点可用 `Thinking` 控制深度思考开关，映射到请求字段 `thinking.type`：
  `Enabled` / `Disabled` 分别对应 `enabled` / `disabled`；`Auto` 不下发 `thinking` 字段，让服务端走默认策略。
- `ReasoningEffort` 映射到 `reasoning_effort`，用于调节思维链长度/强度：`Minimal` / `Low` /
  `Medium` / `High` / `Max`。在火山方舟中，`Minimal` 也表示关闭思考并直接回答；更推荐用
  `Thinking = OpenAIThinkingMode.Disabled` 表达“关闭深度思考”。
- 旧配置 `EnableThinking = false` 仍可用，现在会映射为 `thinking={"type":"disabled"}`；新代码建议直接使用
  `Thinking`。
- 流式协议下 usage 依赖服务端末块，`LlmCompletionResult.TotalTokens` 采用字符数作粗略指标；
  需精确计费请走非流式接口读取 `response.Usage`。
- 音频输入在标准文本协议下不支持，遇到 `AudioContent` 会抛 `NotSupportedException`。

### 关闭或降低思考

火山方舟 Chat API：

```csharp
services.AddOpenAILLMService(o =>
{
    o.ApiKey = "<your-api-key>";
    o.Endpoint = new Uri("https://ark.cn-beijing.volces.com/api/v3");
    o.Model = "doubao-seed-2-0-mini-260428";
    o.Thinking = OpenAIThinkingMode.Disabled; // 下发 thinking={"type":"disabled"}
    o.SurfaceReasoning = false;
});
```

调节思考长度：

```csharp
services.AddOpenAILLMService(o =>
{
    o.ApiKey = "<your-api-key>";
    o.Endpoint = new Uri("https://ark.cn-beijing.volces.com/api/v3");
    o.Model = "deepseek-v4-flash-260425";
    o.ReasoningEffort = OpenAIReasoningEffortLevel.Low; // 下发 reasoning_effort="low"
});
```

## 完整示例

见 `src/Demos/OpenAILLMDemo`（文本流式问答 / 多轮 chat / 一次性补全三种模式）。
