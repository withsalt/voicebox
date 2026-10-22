using System.Collections.Generic;
using LLama.Native;
using VoiceBox.Core.Services.LLMService.Models;

namespace VoiceBox.LLM.LLamaSharp
{
    /// <summary>
    /// LLamaSharp LLM 服务配置。
    ///
    /// <see cref="ModelPath"/> 指向单个 GGUF 文件 (llama.cpp 格式), 任何 llama.cpp 支持的
    /// decoder-only 模型均可加载; chat template 默认取自 GGUF 内嵌的
    /// <c>tokenizer.chat_template</c> 元数据, 特殊模型可通过 <see cref="ModelProfile"/> 定制。
    /// </summary>
    public sealed class LLamaSharpLLMOptions
    {
        /// <summary>
        /// GGUF 模型文件路径 (单文件, 例如 <c>MiniCPM5-1B-Q4_K_M.gguf</c>)。
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// 上下文窗口大小 (token 数)。<c>null</c> 表示沿用 GGUF 训练时的上下文长度
        /// (可能非常大, 会占用大量 KV-cache 内存); 生产环境建议按需显式设置。
        /// 默认 8192, 与 OnnxGenAI 服务的 SessionMaxLength 对齐。
        /// </summary>
        public uint? ContextSize { get; set; } = 8192;

        /// <summary>
        /// 卸载到 GPU 的层数。0 表示纯 CPU; -1 (或大于模型层数的值) 表示全部卸载。
        /// 需要引用对应的 LLamaSharp.Backend.* native 包 (Cuda12/Vulkan 等) 才会生效。
        /// </summary>
        public int GpuLayerCount { get; set; } = 0;

        /// <summary>
        /// 主 GPU 设备号 (多卡时用)。
        /// </summary>
        public int MainGpu { get; set; } = 0;

        /// <summary>
        /// 推理线程数。<c>null</c> 表示由 LLamaSharp 按物理核数自动决定。
        /// </summary>
        public int? Threads { get; set; }

        /// <summary>
        /// prefill 阶段单个 batch 的最大 token 数。prompt 超过该值时分批投喂。
        /// </summary>
        public uint BatchSize { get; set; } = 512;

        /// <summary>
        /// 是否启用 Flash Attention (需 backend 支持; CPU backend 会自动忽略)。
        /// </summary>
        public bool FlashAttention { get; set; } = false;

        /// <summary>
        /// Controls how model weights are loaded. Defaults to <see cref="LLamaLoadMode.MemoryMap"/>
        /// for memory-mapped loading with shared file-backed pages.
        /// Use <see cref="LLamaLoadMode.MemoryLock"/> or <see cref="LLamaLoadMode.MemoryMapAndLock"/>
        /// to keep weights in RAM, or <see cref="LLamaLoadMode.DirectIO"/> for direct I/O when supported.
        /// </summary>
        public LLamaLoadMode LoadMode { get; set; } = LLamaLoadMode.MemoryMap;

        /// <summary>
        /// 同时进行推理的最大并发数。llama.cpp 单 context 下并发请求会争抢同一份计算资源,
        /// 默认 1 以保证延迟稳定; 调高需自行评估内存与吞吐。
        /// </summary>
        public int MaxConcurrentRequests { get; set; } = 1;

        /// <summary>
        /// 默认 <b>新生成</b> token 上限 (语义同 transformers 的 <c>max_new_tokens</c>)。
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
        public float Temperature { get; set; } = 0.7f;

        /// <summary>
        /// Top-K 采样。&lt;=1 时退化为贪心。
        /// </summary>
        public int TopK { get; set; } = 1;

        /// <summary>
        /// Top-P (nucleus) 采样。1.0 表示禁用。
        /// </summary>
        public float TopP { get; set; } = 1.0f;

        /// <summary>
        /// 重复惩罚 (1.0 表示禁用)。作用于最近 <see cref="RepetitionPenaltyWindow"/> 个 token。
        /// </summary>
        public float RepetitionPenalty { get; set; } = 1.0f;

        /// <summary>
        /// 重复惩罚回看的 token 窗口 (语义同 llama.cpp 的 <c>penalty_last_n</c>)。
        /// </summary>
        public int RepetitionPenaltyWindow { get; set; } = 64;

        /// <summary>
        /// 采样随机种子。<c>null</c> 表示不固定 (每次结果可不同);
        /// 设为非负整数可让 <see cref="DoSample"/>=true 时的输出可复现。
        /// </summary>
        public int? RandomSeed { get; set; }

        /// <summary>
        /// 流式输出时, Channel 的有界容量。生成速度远快于消费时, 用此值反压。
        /// 默认 64 块基本不会成为瓶颈; 设为 0 表示无界 (不推荐, 可能 OOM)。
        /// </summary>
        public int StreamChannelCapacity { get; set; } = 64;

        /// <summary>
        /// 是否为 <see cref="Core.Services.LLMService.IConversationalLLMService.CreateConversation"/>
        /// 创建的会话启用跨轮 KV-cache 复用。<c>true</c> (默认) 时, 会话持有独立的 llama.cpp context,
        /// 每轮只对分叉后缀做增量 prefill; <c>false</c> 时会话每轮清空 KV 全量重算,
        /// 作为生产环境的安全阀。每个会话的 context 各占一份 KV-cache 内存
        /// (约 <see cref="ContextSize"/> × 模型 KV 尺寸), 创建大量并发会话时需评估内存。
        /// </summary>
        public bool EnableSessionCache { get; set; } = true;

        /// <summary>
        /// 模型差异扩展点。<c>null</c> (默认) 表示按 GGUF 元数据自动匹配内置 profile
        /// (chat template / thinking 标记 / 附加停止词); 特殊模型可注入自定义实现。
        /// </summary>
        public ILlamaModelProfile? ModelProfile { get; set; }

        /// <summary>
        /// 自定义模型兼容适配器。自适应解析时按添加顺序优先于内置适配器匹配；
        /// <see cref="ModelProfile"/> 非空时仍作为强制选择，优先级最高。
        /// 可通过 <c>options.AdditionalModelProfiles.Add(new MyProfile())</c> 扩展，
        /// 使用依赖注入时也可调用 <c>AddLLamaSharpModelProfile&lt;T&gt;()</c> 注册。
        /// </summary>
        public IList<ILlamaModelProfile> AdditionalModelProfiles { get; } = new List<ILlamaModelProfile>();

        /// <summary>
        /// 构造服务时是否跑一次最小推理预热 (kernel 初始化 / 内存池预分配)。默认 true。
        /// </summary>
        public bool WarmupOnLoad { get; set; } = true;
    }
}
