using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.OnnxGenAI
{
    /// <summary>
    /// LLM 服务配置。
    ///
    /// 模型目录必须是 ONNX Runtime GenAI 兼容格式 (含 <c>genai_config.json</c>);
    /// onnx-community / transformers.js 发布的 ONNX 包**不是**同一种格式, 需通过
    /// onnxruntime-genai 仓库的 <c>builder.py</c> 重新打包 (具体见项目 README)。
    /// </summary>
    public sealed class OnnxGenAILLMOptions
    {
        /// <summary>
        /// 模型目录路径 (包含 <c>genai_config.json</c> + 各 onnx + tokenizer 文件)。
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// 执行提供者。默认 <see cref="GenAIExecutionProvider.FollowConfig"/> 即沿用 genai_config.json。
        /// </summary>
        public GenAIExecutionProvider ExecutionProvider { get; set; } = GenAIExecutionProvider.FollowConfig;

        /// <summary>
        /// GPU 设备号 (Cuda / DirectML 用)。
        /// </summary>
        public int GpuDeviceId { get; set; } = 0;

        /// <summary>
        /// 同时进行推理的最大并发数。ORT-GenAI 的 KV-cache 模型下并发请求会争抢同一份模型权重,
        /// 默认 1 以保证延迟稳定; 调高需自行评估显存/内存。
        /// </summary>
        public int MaxConcurrentRequests { get; set; } = 1;

        /// <summary>
        /// 默认 <b>新生成</b> token 上限 (语义同 transformers 的 <c>max_new_tokens</c>)。
        /// 服务内部会与 prompt token 数相加得到 ORT-GenAI 的 <c>max_length</c>。
        /// 可由每次调用的 <see cref="LlmCompleteOptions.MaxTokens"/> 覆盖。
        /// </summary>
        public int MaxTokens { get; set; } = 4096;

        /// <summary>
        /// 是否启用采样。<c>false</c> 表示贪心解码 (确定性输出)。
        /// </summary>
        public bool DoSample { get; set; } = false;

        /// <summary>
        /// 采样温度 (仅 <see cref="DoSample"/>=true 时生效)。
        /// </summary>
        public float Temperature { get; set; } = 1.0f;

        /// <summary>
        /// Top-K 采样。&lt;=1 时退化为贪心。
        /// </summary>
        public int TopK { get; set; } = 1;

        /// <summary>
        /// Top-P (nucleus) 采样。1.0 表示禁用。
        /// </summary>
        public float TopP { get; set; } = 1.0f;

        /// <summary>
        /// 重复惩罚 (1.0 表示禁用)。
        /// </summary>
        public float RepetitionPenalty { get; set; } = 1.0f;

        /// <summary>
        /// 采样随机种子。<c>null</c> 表示不固定 (每次结果可不同);
        /// 设为非负整数可让 <see cref="DoSample"/>=true 时的输出可复现。
        /// </summary>
        public int? RandomSeed { get; set; }

        /// <summary>
        /// 多模态输入时给 <c>max_length</c> 追加的安全余量 (token 数)。
        ///
        /// 服务用文本 tokenizer 估算 prompt 长度, 但音频/图像会被处理器展开成
        /// 几十 ~ 几百个 soft token, 文本估算并不包含它们。若不补偿, <c>max_length</c>
        /// 会偏小, 新 token 预算被 soft token 挤占, 导致长音频没生成几个字就触顶。
        /// 新 token 的真实上限已由生成循环按计数强制 (见 OnnxGenAILLMService), 此余量仅用于
        /// 放宽 ORT-GenAI 的 <c>max_length</c> 硬上限, 避免它提前停机。默认 512 覆盖常见音频片段;
        /// 处理超长音频时可调大。仅在存在音频/图像输入时生效。
        /// </summary>
        public int MaxLengthMediaSafetyMargin { get; set; } = 512;

        /// <summary>
        /// 流式输出时, Channel 的有界容量。生成速度远快于消费时, 用此值反压。
        /// 默认 64 块基本不会成为瓶颈; 设为 0 表示无界 (不推荐, 可能 OOM)。
        /// </summary>
        public int StreamChannelCapacity { get; set; } = 64;

        /// <summary>
        /// 是否为 <see cref="Core.Services.LLMService.IConversationalLLMService.CreateConversation"/>
        /// 创建的会话启用跨轮 KV-cache 复用。<c>true</c> (默认) 时, 会话持有一个长期存活的 Generator,
        /// 每轮只对分叉后缀做增量 prefill; <c>false</c> 时会话退化为每轮新建 Generator (等价无状态路径),
        /// 作为生产环境的安全阀。
        ///
        /// 注意: 复用仅在 <see cref="MaxConcurrentRequests"/> == 1 时生效 (KV-cache 复用要求单请求串行);
        /// 含音频/图像的轮次、或采样参数发生变化的轮次会自动回退到一次性 Generator。
        /// </summary>
        public bool EnableSessionCache { get; set; } = true;

        /// <summary>
        /// 会话持久 Generator 的 <c>max_length</c> 上限 (token 总数)。会话跨轮累积, 该上限须能容纳
        /// 整段对话 (prompt + 生成); 触顶即停 (上下文满)。per-turn 的新 token 上限仍由 MaxTokens 单独强制。
        /// 默认 8192, 按模型上下文窗口与预期对话长度调整。仅影响会话路径, 不影响无状态调用。
        /// </summary>
        public int SessionMaxLength { get; set; } = 8192;
    }
}
