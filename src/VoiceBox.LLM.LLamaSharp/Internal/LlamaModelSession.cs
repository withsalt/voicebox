using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using LLama;
using LLama.Common;
using LLama.Native;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    /// <summary>
    /// 持有单个 GGUF 模型的全部本地资源: <see cref="Weights"/> (共享权重) 与
    /// <see cref="SharedContext"/> (无状态路径复用的常驻 context)。
    ///
    /// 这一层只负责 "校验文件 / 加载 / 元数据解析 / profile 解析 / 预热 / Dispose";
    /// 不感知 message 结构, 也不感知流式/非流式调用语义, 那些在 <c>LLamaSharpLLMService</c> 里组装。
    /// </summary>
    internal sealed class LlamaModelSession : IDisposable
    {
        private readonly ModelParams _modelParams;
        private int _disposed;

        public LlamaModelSession(
            LLamaSharpLLMOptions options,
            IEnumerable<ILlamaModelProfile>? injectedProfiles = null)
        {
            ValidateModelFile(options.ModelPath);

            _modelParams = BuildModelParams(options);

            Weights = LLamaWeights.LoadFromFile(_modelParams);
            try
            {
                ModelInfo = new LlamaModelInfo(Weights.Metadata);
                IEnumerable<ILlamaModelProfile> profiles = options.AdditionalModelProfiles;
                if (injectedProfiles is not null)
                    profiles = profiles.Concat(injectedProfiles);
                Profile = LlamaModelProfileRegistry.Resolve(ModelInfo, options.ModelProfile, profiles);
                DecoderConfig = new LlamaStreamDecoderConfig(Profile);
                Encoding = _modelParams.Encoding;
                BatchSize = checked((int)options.BatchSize);

                SharedContext = Weights.CreateContext(_modelParams);
            }
            catch
            {
                Weights.Dispose();
                throw;
            }
        }

        public LLamaWeights Weights { get; }

        /// <summary>
        /// 无状态调用路径复用的常驻 context (每次调用前清空 KV)。
        /// 仅可在持有服务并发闸门时使用; 会话路径各自创建独立 context。
        /// </summary>
        public LLamaContext SharedContext { get; }

        public LlamaModelInfo ModelInfo { get; }

        public ILlamaModelProfile Profile { get; }

        public LlamaStreamDecoderConfig DecoderConfig { get; }

        public Encoding Encoding { get; }

        /// <summary>prefill 阶段单个 batch 的最大 token 数。</summary>
        public int BatchSize { get; }

        /// <summary>为一段会话创建独立 context (独立 KV-cache), 由调用方 Dispose。</summary>
        public LLamaContext CreateConversationContext() => Weights.CreateContext(_modelParams);

        /// <summary>
        /// 跑一次最小推理 (1 个 prompt token + 1 步 decode), 让 backend 完成线程池/
        /// 计算图初始化与内存池预分配, 之后清空 KV。失败由调用方记录, 不影响后续可用性。
        /// </summary>
        public void Warmup()
        {
            var tokens = SharedContext.Tokenize("hi", addBos: true, special: true);
            if (tokens.Length == 0) return;

            var batch = new LLamaBatch();
            for (int i = 0; i < tokens.Length; i++)
                batch.Add(tokens[i], i, LLamaSeqId.Zero, logits: i == tokens.Length - 1);
            try
            {
                LlamaGeneration.Decode(SharedContext, batch);
            }
            finally
            {
                SharedContext.NativeHandle.MemoryClear(true);
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            // 反向释放: context 依赖 weights。
            SharedContext.Dispose();
            Weights.Dispose();
        }

        private static ModelParams BuildModelParams(LLamaSharpLLMOptions options)
        {
            var p = new ModelParams(options.ModelPath)
            {
                ContextSize = options.ContextSize,
                // llama.cpp 语义: 卸载层数超过模型层数即全部卸载; -1 按惯例表示"全部"。
                GpuLayerCount = options.GpuLayerCount < 0 ? int.MaxValue : options.GpuLayerCount,
                MainGpu = options.MainGpu,
                BatchSize = options.BatchSize,
                FlashAttention = options.FlashAttention,
                LoadMode = options.LoadMode,
            };
            if (options.Threads is int threads)
            {
                p.Threads = threads;
                p.BatchThreads = threads;
            }
            return p;
        }

        private static void ValidateModelFile(string modelPath)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentException("ModelPath cannot be empty.", nameof(modelPath));
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"GGUF model file not found: {modelPath}", modelPath);

            // GGUF magic: 0x47 0x47 0x55 0x46 ("GGUF")。提前给出清晰报错,
            // 避免 native 层加载失败时只有一条含糊的 llama.cpp 日志。
            Span<byte> magic = stackalloc byte[4];
            using (var fs = File.OpenRead(modelPath))
            {
                if (fs.Read(magic) != 4 || magic[0] != (byte)'G' || magic[1] != (byte)'G' || magic[2] != (byte)'U' || magic[3] != (byte)'F')
                {
                    throw new InvalidDataException(
                        $"'{modelPath}' is not a GGUF file (bad magic). " +
                        "VoiceBox.LLM.LLamaSharp loads llama.cpp GGUF models only — " +
                        "ONNX / safetensors checkpoints must be converted with llama.cpp's convert scripts first.");
                }
            }
        }
    }
}
