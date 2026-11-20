# VoiceBox 项目约定

本文件面向所有 AI 编码助手 / 贡献者。修改 C# 代码时请遵循以下规则。

## C# 换行规则

### 核心规则

- 多行构造合并后，若整行（含缩进、行尾 `;` / `{` / `: base(...)` 等）**不超过 140 个字符**，必须写成单行。
- 超过 140 个字符才允许换行；需要换行时，每项一行。
- 行长按字符数计算（中文等宽字符按 1 个字符计）。

### 必须合并的构造

1. **参数列表**：方法、构造器、record primary constructor、委托、局部函数、索引器。
2. **实参列表**：方法调用、`new` 对象创建、特性参数、`: base(...)` / `: this(...)`。
3. **其它列表**：base 类型列表、泛型类型参数/实参、元组。
4. **表达式续行**：
   - `=` 后的值换行
   - `=>` 后的表达式换行
   - 二元表达式（`&&`、`||`、`??`、`+` 等）
   - 三元表达式
   - lambda 表达式

### 禁止合并（允许保持换行）

- **对象/集合/数组初始化器块**：`new Foo { ... }`、`[...]`、`new[] { ... }`、`with { ... }`。
  - 即使合并后 ≤140 字符，也保持多行。
  - 外层表达式（`=`、调用实参、lambda 等）跨越这些块时，也不得合并。
- **链式调用续行**：以 `.` 开头的行（如 `.AddFoo()`）保持原样。
- **无法安全合并的构造**：含多行 lambda、多行字符串（raw/verbatim）或注释时保持原样。
- 多行初始化器保留尾逗号；合并成单行的列表去掉闭合符前的尾逗号。

### 示例

应合并为单行（≤140 字符）：

```csharp
public ConversationToolInvocation(string callId, JsonElement arguments, ConversationToolExecutionPreference executionPreference)

throw new ConversationProtocolException("The conversation emitted thinking after its completed update.");

_runtime = AssistantRuntimeFactory.Create(composition, options, logger, eventSender: this);
```

允许保持换行（初始化器块）：

```csharp
var toolOptions = new ConversationToolOptions
{
    MaxParallelToolCalls = 2,
    ToolTranscriptRetentionTurns = 3,
};

KeyValuePair<string, string>[] parameters =
[
    new("coor", "bd09ll"),
    new("ak", Ak),
];
```

合并后超过 140 字符的参数/实参列表，保持每项一行。
