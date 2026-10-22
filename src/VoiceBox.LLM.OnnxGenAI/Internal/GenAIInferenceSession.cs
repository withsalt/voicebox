using System.IO;
using Microsoft.ML.OnnxRuntimeGenAI;
using VoiceBox.Core.Onnx;

namespace VoiceBox.LLM.OnnxGenAI.Internal
{
    /// <summary>
    /// 持有单个 ORT-GenAI 模型的全部本地资源:
    /// <see cref="Model"/> / <see cref="Tokenizer"/> / <see cref="MultiModalProcessor"/>。
    ///
    /// 这一层只负责 "构造 / Dispose / 校验目录格式 / 预热"; 不感知 message 结构,
    /// 也不感知流式/非流式调用语义, 那些在 <c>OnnxGenAILLMService</c> 里组装。
    /// </summary>
    internal sealed class GenAIInferenceSession : System.IDisposable
    {
        private readonly Config _config;
        private int _disposed;

        public GenAIInferenceSession(OnnxGenAILLMOptions options)
        {
            ValidateModelDirectory(options.ModelPath);

            _config = new Config(options.ModelPath);

            ConfigureExecutionProvider(_config, options);

            Model = new Model(_config);
            Tokenizer = new Tokenizer(Model);
            // MultiModalProcessor 仅在 genai_config.json 含 vision/speech 段时才能成功创建。
            // 对于 text-only 模型这里会抛错; 我们先尝试创建, 失败时退回到纯文本路径。
            try
            {
                Processor = new MultiModalProcessor(Model);
            }
            catch (OnnxRuntimeGenAIException)
            {
                Processor = null;
            }
        }

        public Model Model { get; }

        public Tokenizer Tokenizer { get; }

        /// <summary>
        /// 多模态处理器。<c>null</c> 表示当前模型只支持纯文本输入 (无 vision/speech 段)。
        /// </summary>
        public MultiModalProcessor? Processor { get; }

        /// <summary>
        /// 跑一次最小推理, 让 ORT 完成 kernel JIT / 内存池预分配。失败仅记录, 不影响后续可用性。
        /// </summary>
        public void Warmup()
        {
            using var sequences = Tokenizer.Encode("hi");
            using var generatorParams = new GeneratorParams(Model);
            generatorParams.SetSearchOption("max_length", (double)sequences[0].Length + 1);
            generatorParams.SetSearchOption("do_sample", false);

            using var generator = new Generator(Model, generatorParams);
            generator.AppendTokenSequences(sequences);
            generator.GenerateNextToken();
        }

        public void Dispose()
        {
            if (System.Threading.Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            // 反向释放: Processor 与 Tokenizer 依赖 Model, Model 依赖 Config。
            Processor?.Dispose();
            Tokenizer.Dispose();
            Model.Dispose();
            _config.Dispose();
        }

        private static void ValidateModelDirectory(string modelPath)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new System.ArgumentException("ModelPath cannot be empty.", nameof(modelPath));
            if (!Directory.Exists(modelPath))
                throw new DirectoryNotFoundException($"Model directory not found: {modelPath}");

            string genaiConfigPath = Path.Combine(modelPath, "genai_config.json");
            if (!File.Exists(genaiConfigPath))
            {
                throw new FileNotFoundException(
                    $"genai_config.json not found in '{modelPath}'. " +
                    "This directory does not appear to be an ONNX Runtime GenAI model package. " +
                    "Note that onnx-community / transformers.js ONNX exports are NOT compatible — " +
                    "you must rebuild the model using onnxruntime-genai's model builder. " +
                    "See VoiceBox.LLM.OnnxGenAI/README.md for instructions.",
                    genaiConfigPath);
            }
        }

        private static void ConfigureExecutionProvider(Config config, OnnxGenAILLMOptions options)
        {
            switch (options.ExecutionProvider)
            {
                case GenAIExecutionProvider.FollowConfig:
                    // 不动 config 中已有的 provider 设置。
                    break;

                case GenAIExecutionProvider.Cpu:
                    config.ClearProviders();
                    //config.AppendProvider("cpu");
                    break;

                case GenAIExecutionProvider.Cuda:
                    config.ClearProviders();
                    config.AppendProvider("cuda");
                    config.SetProviderOption("cuda", "device_id", options.GpuDeviceId.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    break;

                case GenAIExecutionProvider.DirectML:
                    config.ClearProviders();
                    config.AppendProvider("dml");
                    config.SetProviderOption("dml", "device_id", options.GpuDeviceId.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    break;

                default:
                    throw new System.ArgumentOutOfRangeException(
                        nameof(options.ExecutionProvider), options.ExecutionProvider,
                        "Unknown execution provider.");
            }
        }
    }
}
