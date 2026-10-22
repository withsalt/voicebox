# VoiceBox.LLM.OnnxGenAI

基于 [onnxruntime-genai](https://github.com/microsoft/onnxruntime-genai) 的**本地**多模态 LLM 推理实现,
与 VoiceBox 其它服务 (TTS / ASR / VAD / DNR) 同构, 实现 `ILLMService` / `IStreamLLMService` 接口。

支持纯文本、以及音频/图像输入 (取决于模型自身能力, 例如 Gemma 3 / Holo 等多模态模型)。模型加载、
KV-cache、GPU 调度全部在进程内完成 — 与远端 `VoiceBox.LLM.OpenAI` 相对,
这里适合离线、单机、低延迟、数据不出机的场景。

## 模型格式 (务必先读)

模型目录**必须**是 ONNX Runtime GenAI 兼容格式, 即包含 `genai_config.json` + 各 `*.onnx` +
tokenizer 文件 (以及多模态模型的 `chat_template.jinja`)。

> ⚠️ **onnx-community / transformers.js 发布的 ONNX 包不是这种格式**, 无法直接加载。
> 需通过 onnxruntime-genai 的 model builder 重新打包:
>
> ```bash
> pip install onnxruntime-genai
> python -m onnxruntime_genai.models.builder \
>     -m <hf-model-id-or-path> \
>     -o <output-dir> \
>     -p int4 \
>     -e cpu            # 或 cuda / dml
> ```
>
> 打包后目录里应能看到 `genai_config.json`, 用它作为 `ModelPath`。

## 关键依赖

| 包 | 用途 |
|---|---|
| `Microsoft.ML.OnnxRuntimeGenAI.Managed` | 托管封装 (本项目直接引用) |
| `Microsoft.ML.OnnxRuntimeGenAI` / `.Cuda` / `.DirectML` | 原生运行时, 按执行提供者由**宿主**工程引用 |
| `Microsoft.Extensions.Options` / `.Logging.Abstractions` | 配置与日志契约 |

> Managed 包只含托管封装, 不含原生 `.dll/.so`。宿主工程需按目标 EP 额外引用对应原生包
> (CPU: `Microsoft.ML.OnnxRuntimeGenAI`; NVIDIA: `.Cuda`; Windows GPU: `.DirectML`)。

## 用法

### 直接构造

```csharp
using Microsoft.Extensions.Options;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.LLM.OnnxGenAI;

var options = Options.Create(new OnnxGenAILLMOptions
{
    ModelPath         = @"E:\models\gemma-3-4b\cpu",  // 含 genai_config.json
    ExecutionProvider = GenAIExecutionProvider.Cpu,
    MaxTokens         = 4096,
    DoSample          = true,
    Temperature       = 0.5f,
    TopK              = 40,
    TopP              = 0.9f,
    RepetitionPenalty = 1.1f,
});

using var llm = new OnnxGenAILLMService(options);

// 单轮文本流式
await foreach (var chunk in llm.StreamAsync(new[] { ChatMessage.User("你好, 介绍一下你自己") }))
{
    if (chunk.Kind == LlmChunkKind.Text)
        Console.Write(chunk.Text);
}

// 一次性返回 (内部仍按 token 生成; 已在线程池执行, 不阻塞调用线程)
var result = await llm.CompleteAsync(new[] { ChatMessage.User("光速是多少?") });
Console.WriteLine($"{result.Text}  ({result.CompletionTokens} tok / {result.ElapsedSeconds:F2}s)");
```

### 音频理解 / 转写 (多模态模型)

```csharp
var messages = new[]
{
    ChatMessage.User(
        new TextContent("Transcribe this audio."),
        new AudioContent(@"D:\samples\speech.wav")),   // 也支持 new AudioContent(byte[])
};

await foreach (var chunk in llm.StreamAsync(messages))
    if (chunk.Kind == LlmChunkKind.Text)
        Console.Write(chunk.Text);
```

### 思考模式

```csharp
var r = await llm.CompleteAsync(
    new[] { ChatMessage.User("9.11 和 9.9 哪个更大?") },
    new LlmCompleteOptions { EnableThinking = true });

Console.WriteLine($"思考: {r.Thinking}");
Console.WriteLine($"答案: {r.Text}");
```

流式接口下 `LlmTextChunk.Kind` 区分 `Thinking` 与 `Text`; 流末尾补发一个空的 `IsFinal = true` 标记块
(与其他流式 LLM 服务一致, 供下游收尾)。

### 通过 DI

```csharp
services.AddOnnxGenAILLMService(o =>
{
    o.ModelPath = @"E:\models\gemma-3-4b\cpu";
    o.ExecutionProvider = GenAIExecutionProvider.Cpu;
});

var llm = sp.GetRequiredService<IStreamLLMService>();
```

也可挂到 `AudioServiceManager` 上, 与 Qwen3 / Kokoro 风格一致:

```csharp
var manager = new AudioServiceManager()
    .AddOnnxGenAILLMService(o =>
    {
        o.ModelPath = @"E:\models\gemma-3-4b\cpu";
    });
manager.Build();
```

### 有状态会话 (跨轮 KV-cache 复用)

多轮对话时, 用 `IConversationalLLMService.CreateConversation()` 创建一段会话, 会话内**默认复用**
KV-cache —— 每轮只对与上轮分叉的后缀做增量 prefill, 相同的 system + 既有对话不再重复计算。

调用方式与无状态接口一致: 每轮仍传入<b>完整</b>消息列表 (含 system 与全部历史), 会话内部自行对账。
因此裁剪历史、回滚未回答的 user 消息等操作, 只需照常改变下一轮传入的列表即可, 无需通知会话。

```csharp
using var conversation = ((IConversationalLLMService)llm).CreateConversation();

var history = new List<ChatMessage> { ChatMessage.System("You are concise.") };
history.Add(ChatMessage.User("讲个冷笑话"));

var reply = new System.Text.StringBuilder();
await foreach (var chunk in conversation.StreamAsync(history))
    if (chunk.Kind == LlmChunkKind.Text) reply.Append(chunk.Text);
history.Add(ChatMessage.Assistant(reply.ToString()));

history.Add(ChatMessage.User("再来一个"));      // 这一轮只增量处理新增部分, system + 上一轮命中复用
await foreach (var chunk in conversation.StreamAsync(history)) { /* ... */ }
```

- 会话**非线程安全**: 同一实例的调用必须串行 (一轮结束再起下一轮), 用完 `Dispose`。
- 每段会话独立持有自己的持久 `Generator` (含一份 KV-cache 显存/内存), 互不影响。
- 复用仅在 `MaxConcurrentRequests == 1` 且 `EnableSessionCache=true` 时生效; 含音频/图像的轮次、
  或采样参数发生变化的轮次会自动回退到一次性 Generator (该轮全量 prefill), 输出与无状态路径一致。
- 远端实现 (`VoiceBox.LLM.OpenAI`) 不实现 `IConversationalLLMService`,
  调用方应 `llm as IConversationalLLMService` 判空后优雅回退到 `StreamAsync`。

## 选项

见 `OnnxGenAILLMOptions.cs` 的 XML 注释。重点字段:

| 字段 | 默认值 | 含义 |
|---|---|---|
| `ModelPath` | (必填) | ORT-GenAI 模型目录 (含 `genai_config.json`) |
| `ExecutionProvider` | `FollowConfig` | `FollowConfig` / `Cpu` / `Cuda` / `DirectML` |
| `GpuDeviceId` | `0` | CUDA / DirectML 设备号 |
| `MaxConcurrentRequests` | `1` | 进程内并发上限 (反压; KV-cache 模型下不建议调高) |
| `MaxTokens` | `4096` | 默认**新生成** token 上限 (语义同 `max_new_tokens`) |
| `DoSample` / `Temperature` / `TopK` / `TopP` | `false`/1.0/1/1.0 | 采样参数 |
| `RepetitionPenalty` | `1.0` | 重复惩罚 (1.0 = 禁用) |
| `RandomSeed` | `null` | 采样随机种子; 设非负整数使 `DoSample=true` 时可复现 |
| `MaxLengthMediaSafetyMargin` | `512` | 多模态下给 `max_length` 追加的余量 (补偿音频/图像 soft token) |
| `StreamChannelCapacity` | `64` | 流式反压阈值; 0 = 无界 (不推荐) |
| `EnableSessionCache` | `true` | 会话是否跨轮复用 KV-cache; `false` 时会话每轮新建 Generator (安全阀) |
| `SessionMaxLength` | `8192` | 会话持久 Generator 的 `max_length` 上限, 须容纳整段对话; 仅影响会话路径 |

每次调用还可用 `LlmCompleteOptions` 覆盖 `MaxTokens` / 采样参数 / `EnableThinking`。

## 设计

```
OnnxGenAILLMService                 (公共: 实现 IConversationalLLMService : IStreamLLMService)
   │
   ├── GenAIInferenceSession    (Internal: Model / Tokenizer / MultiModalProcessor 的构造与释放)
   ├── GenAIPromptTemplate      (Internal: ChatMessage → chat_template.jinja 渲染)
   ├── GenAIInputBuilder        (Internal: 拆解音频/图像为 ORT-GenAI 输入对象)
   ├── GenAIStreamDecoder       (Internal: 逐 token 分流 thinking / 过滤 special token)
   ├── GenAIGeneration          (Internal: 解码循环/一次性生成/前缀比对等可复用内核)
   └── OnnxGenAILLMConversation     (Internal: 有状态会话, token-LCP + RewindTo 复用 KV-cache)
```

- **生命周期**: 构造期完成全部重资源加载与一次预热, 之后调用零冷启动开销; 按单例注册。
- **并发**: `SemaphoreSlim` 串行化 `Generator` 的创建与运行 (同一 Model 上多并发会拖垮 KV-cache)。
- **流式**: `Channel<LlmTextChunk>` 解耦生产/消费, `StreamChannelCapacity` 提供反压。
- **非流式**: `CompleteAsync` 在线程池执行生成循环, 不把阻塞式推理压在调用线程上。
- **新 token 上限**: 由生成循环按计数权威强制, `max_length` 只作 ORT 侧兜底 — 使多模态下
  `MaxTokens` 语义准确 (文本 tokenizer 无法预估音频/图像展开的 soft token 数)。
- **跨轮复用**: 会话路径用 token-level 最长公共前缀 + `RewindTo`/`AppendTokens` 增量 prefill;
  复用后 KV 序列恒等于本轮完整 prompt, 与新建 Generator 完全等价 (复用只省 prefill, 不改输出)。
- **取消**: 全程 `CancellationToken`; 已发出的块仍可被消费者读完, 之后通道抛 `OperationCanceledException`。

## 已知限制

1. **多模态一次一段音频**: Gemma 4 处理器每 prompt 最多 1 个音频片段, 超出抛 `NotSupportedException`。
2. **无状态路径不复用**: `StreamAsync`/`CompleteAsync` 每次新建 Generator; 跨轮复用需用
   `CreateConversation()` 会话 (见上文)。会话内历史中段被裁剪时, 只能复用到 system 前缀, 尾部重 prefill。
3. **无批处理**: `MaxConcurrentRequests` 只做串行反压, 未利用 batched generation。
4. **停止序列 (stop sequences) 未暴露**: 目前仅靠 `MaxTokens` 与模型自身 EOS 收敛。

## 参考

- onnxruntime-genai: <https://github.com/microsoft/onnxruntime-genai>
- Model builder: <https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builder.py>
- 同接口的远端实现: `VoiceBox.LLM.OpenAI`
