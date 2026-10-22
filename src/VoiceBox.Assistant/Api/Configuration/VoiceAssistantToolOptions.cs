namespace VoiceBox.Assistant.Api.Configuration
{
    /// <summary>
    /// 结构化工具调用的运行参数。只在通过集成程序集安装了工具能力时生效; 未启用工具时这些值
    /// 不参与任何决策, text-only 路径的行为与不存在该配置节时完全一致。
    /// </summary>
    public sealed class VoiceAssistantToolOptions
    {
        /// <summary>
        /// 最近多少轮对话在模型 prompt 中保留 tool call / tool result 消息。更早的轮次退化为
        /// 纯 user/assistant 文本, 使模型仍能看到对话脉络但不再携带结构化工具数据。
        /// 这是一个上下文预算闸门: MCP 结果单条可达 <c>MaxResultBytes</c> (默认 1MB),
        /// 无限保留会在长会话中撑爆 context。
        /// 取 0 表示完全不持久化工具消息 (等价于引入该选项之前的行为)。
        /// </summary>
        public int ToolTranscriptRetentionTurns { get; set; } = 3;

        /// <summary>
        /// 单条 tool result 落库时的字节上限。超出的结果会被截断并标注, 模型据此知道内容不完整,
        /// 而不是收到一段被静默切断的 JSON。仅影响持久化副本; 产生该结果的那一轮模型续推始终
        /// 看到完整结果。
        /// </summary>
        public int MaxPersistedToolResultBytes { get; set; } = 8 * 1024;

        /// <summary>
        /// 同一批 provider tool call 的最大并发执行数。只有 <c>ReadOnly</c> / <c>Sensitive</c>
        /// 风险的工具参与并发; 有副作用的工具始终按 provider 调用顺序串行, 使外部副作用顺序确定。
        /// 取 1 表示全部串行 (等价于引入该选项之前的行为)。
        /// </summary>
        public int MaxParallelToolCalls { get; set; } = 4;

        /// <summary>
        /// 瞬时失败的最大重试次数。只有 <c>ReadOnly</c> 工具且失败被判定为可重试 (网络异常、
        /// 5xx、429、传输超时) 时才重试; <c>SideEffecting</c> 及以上风险永不自动重试。
        /// 取 0 表示关闭重试。
        /// </summary>
        public int MaxToolRetryAttempts { get; set; } = 2;

        /// <summary>
        /// 首次重试前的退避时长, 之后每次乘 3 (默认 200ms / 600ms)。
        /// </summary>
        public int ToolRetryBaseDelayMs { get; set; } = 200;

        /// <summary>
        /// 单个工具调用含全部重试的总耗时上限。与 <c>Timeout × (1 + MaxToolRetryAttempts)</c>
        /// 取较小值, 防止一个慢工具用重试把整轮语音回复拖到不可接受的时长。
        /// </summary>
        public int MaxToolRetryBudgetMs { get; set; } = 60_000;

        /// <summary>
        /// 是否在宿主侧按工具的 input JSON Schema 校验模型给出的参数。开启后非法参数不会到达
        /// 工具实现, 而是以 <c>InputRequired</c> + 结构化违规明细回灌模型, 让模型自行修正。
        /// </summary>
        public bool ValidateToolArguments { get; set; } = true;
    }
}
