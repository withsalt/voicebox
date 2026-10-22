using System.Globalization;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.DNR.Hush.Internal;
using VoiceBox.DNR.Hush.Models;

namespace VoiceBox.DNR.Hush
{
    /// <summary>
    /// 基于 Hush (DeepFilterNet3-SE, 16 kHz) 的实时数字降噪服务。
    ///
    /// 工作管线：
    ///   input(16 kHz, mono, f32)
    ///     → 按 hop=160 (10ms) 分帧：STFT → ERB/Spec 特征(运行归一化) → 有状态 ONNX(三 GRU 态循环)
    ///       → ERB 掩码 + order-5 深度滤波 → ISTFT 重叠相加
    ///     → 写回 buffer (剩余样本保留至下次调用)。
    /// 固有算法延迟 = (fft−hop) + lookahead·hop（Hush 为 160 样本=10ms）。与
    /// <see cref="IDigitalNoiseReductionService"/> 一致，单次调用最多回写 <c>buffer.Length</c> 个样本。
    /// </summary>
    public sealed class HushDNRService : IHushDNRService
    {
        private readonly HushOptions _options;
        private readonly int _hopSize;
        private readonly int _delaySamplesModelRate;

        private HushProcessor? _processor;

        // 16k 域：不足一帧的累积尾部、16k 降噪输出累积。
        private readonly float[] _frameAccum;   // 长度 hop
        private int _frameAccumCount;
        private float[] _denoised16k = [];
        private readonly float[] _frameOut;     // 长度 hop

        // 输入采样率域的待输出环形缓冲。
        private float[] _pendingOutput = [];
        private int _pendingHead;
        private int _pendingCount;

        // 跨流仅在起始丢弃算法延迟对应的 16k 输出样本。
        private int _delaySkipRemaining;

        private readonly Lock _lock = new();
        private bool _disposed;

        /// <summary>输入采样率下一帧 (10 ms) 的样本数，供调用方设置缓冲尺寸参考。实际 Process 接受任意长度。</summary>
        public int FrameSize { get; }
        public int DelaySamples { get; }

        public HushDNRService(IOptions<HushOptions> options)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));

            if (_options.ChannelCount != 1)
                throw new NotSupportedException($"HushDNRService only supports mono audio (ChannelCount=1), got {_options.ChannelCount}.");
            if (_options.SampleFormat != SampleFormat.Float32)
                throw new NotSupportedException($"HushDNRService only supports Float32, got {_options.SampleFormat}.");
            if (_options.SampleRate != 16000)
                throw new NotSupportedException($"HushDNRService only supports 16000 Hz audio, got {_options.SampleRate} Hz.");

            string modelDir = ResolveModelDir(_options);
            string modelPath = Path.Combine(modelDir, "combined_streaming.onnx");
            if (!File.Exists(modelPath))
                throw new FileNotFoundException(
                    $"Hush streaming model not found: {modelPath}. " +
                    "Expected a 'combined_streaming.onnx' with externalized GRU states.", modelPath);
            string configPath = Path.Combine(modelDir, "config.ini");

            var cfg = ParseIni(configPath);
            int sr = GetInt(cfg, "sr", 16000);
            int fftSize = GetInt(cfg, "fft_size", 320);
            int hopSize = GetInt(cfg, "hop_size", 160);
            int nbErb = GetInt(cfg, "nb_erb", 32);
            int nbDf = GetInt(cfg, "nb_df", 64);
            int dfOrder = GetInt(cfg, "df_order", 5);
            int dfLookahead = GetInt(cfg, "df_lookahead", 0);
            int convLookahead = GetInt(cfg, "conv_lookahead", 0);
            int minNbErbFreqs = GetInt(cfg, "min_nb_erb_freqs", 2);
            float normTau = GetFloat(cfg, "norm_tau", 1f);
            int encKernelT = GetCsvFirstInt(cfg, "conv_kernel_inp", 3);
            if (dfLookahead < 0) throw new InvalidOperationException($"Invalid config: df_lookahead={dfLookahead} (must be >= 0). [{configPath}]");
            if (convLookahead < 0) throw new InvalidOperationException($"Invalid config: conv_lookahead={convLookahead} (must be >= 0). [{configPath}]");
            if (encKernelT <= 0) throw new InvalidOperationException($"Invalid config: conv_kernel_inp first value={encKernelT} (must be > 0). [{configPath}]");
            int lookahead = Math.Max(dfLookahead, convLookahead);
            int encWindow = _options.EncoderContextFrames ?? encKernelT;
            // encWindow 必须 >= 编码器时间卷积核大小：小于它会截断因果感受野，
            // 使流式输出与离线模型静默不一致（增大则安全，只是多算上下文）。
            if (encWindow < encKernelT)
                throw new ArgumentException(
                    $"EncoderContextFrames ({encWindow}) must be >= the encoder temporal kernel ({encKernelT}); " +
                    "a smaller context silently truncates the causal receptive field and corrupts output.",
                    nameof(options));

            // 尽早校验从 config.ini 读到的 DSP 参数，避免损坏/自定义配置导致后续崩溃或构造期挂起
            // （例如 hop=0、norm_tau<=0 会破坏归一化系数计算）。
            if (sr <= 0) throw new InvalidOperationException($"Invalid config: sr={sr} (must be > 0). [{configPath}]");
            if (fftSize <= 0 || (fftSize & 1) != 0) throw new InvalidOperationException($"Invalid config: fft_size={fftSize} (must be positive even). [{configPath}]");
            if (hopSize <= 0 || hopSize * 2 > fftSize) throw new InvalidOperationException($"Invalid config: hop_size={hopSize} (must be > 0 and hop*2 <= fft_size={fftSize}). [{configPath}]");
            if (nbErb < 2) throw new InvalidOperationException($"Invalid config: nb_erb={nbErb} (must be >= 2). [{configPath}]");
            if (nbDf < 2 || nbDf > fftSize / 2 + 1) throw new InvalidOperationException($"Invalid config: nb_df={nbDf} (must be in [2, {fftSize / 2 + 1}]). [{configPath}]");
            if (dfOrder <= 0) throw new InvalidOperationException($"Invalid config: df_order={dfOrder} (must be > 0). [{configPath}]");
            if (minNbErbFreqs <= 0) throw new InvalidOperationException($"Invalid config: min_nb_erb_freqs={minNbErbFreqs} (must be > 0). [{configPath}]");
            if (!(normTau > 0f)) throw new InvalidOperationException($"Invalid config: norm_tau={normTau} (must be > 0). [{configPath}]");
            if (sr != 16000) throw new InvalidOperationException($"Invalid config: sr={sr} (HushDNRService only supports 16000). [{configPath}]");

            _hopSize = hopSize;
            _delaySamplesModelRate = (fftSize - hopSize) + lookahead * hopSize;

            _processor = new HushProcessor(
                modelPath, _options, sr, fftSize, hopSize, nbErb, nbDf, dfOrder,
                lookahead, minNbErbFreqs, normTau, encWindow);

            _frameAccum = new float[_hopSize];
            _frameOut = new float[_hopSize];

            FrameSize = _hopSize;
            DelaySamples = _delaySamplesModelRate;
            _delaySkipRemaining = _delaySamplesModelRate;
        }

        private static string ResolveModelDir(HushOptions options)
        {
            if (!string.IsNullOrWhiteSpace(options.ModelDir))
            {
                if (!Directory.Exists(options.ModelDir))
                    throw new DirectoryNotFoundException($"Hush model directory not found: {options.ModelDir}");
                return options.ModelDir;
            }

            string dir = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "dnr", "hush", options.Model.ToDirName());
            if (!Directory.Exists(dir))
                throw new DirectoryNotFoundException(
                    $"Built-in Hush model '{options.Model}' not found at '{dir}'. " +
                    "Ensure the model files are copied to the output directory, or set HushOptions.ModelDir.");
            return dir;
        }

        public int Process(Span<float> buffer)
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_processor is null)
                    throw new InvalidOperationException("Hush processor is not initialized.");

                int inputLen = buffer.Length;

                // 1) 16k 流式分帧降噪 → _denoised16k。
                int produced16k = ProcessFrames(buffer);

                // 1b) 延迟补偿：起始丢弃 delaySamples 个 16k 输出。
                int off16k = 0;
                if (_delaySkipRemaining > 0 && produced16k > 0)
                {
                    int skip = Math.Min(_delaySkipRemaining, produced16k);
                    off16k += skip;
                    produced16k -= skip;
                    _delaySkipRemaining -= skip;
                }

                // 2) 16k 降噪输出 → pending。
                if (produced16k > 0)
                    AppendPending(_denoised16k.AsSpan(off16k, produced16k));

                // 3) 回写 (最多 inputLen)。
                int toCopy = Math.Min(_pendingCount, inputLen);
                if (toCopy > 0) DrainPending(buffer, toCopy);

                return toCopy;
            }
        }

        /// <summary>
        /// 将 16k <paramref name="samples"/> 按 hop 分帧处理，输出写入 <see cref="_denoised16k"/>，返回样本数。
        /// </summary>
        private int ProcessFrames(ReadOnlySpan<float> samples)
        {
            int hop = _hopSize;
            int maxFrames = (samples.Length + _frameAccumCount) / hop;
            EnsureCapacity(ref _denoised16k, Math.Max(1, maxFrames * hop));

            int outLen = 0;
            int pos = 0;

            // 先补满上次遗留的不完整帧。
            if (_frameAccumCount > 0)
            {
                int need = hop - _frameAccumCount;
                int take = Math.Min(need, samples.Length);
                samples.Slice(0, take).CopyTo(_frameAccum.AsSpan(_frameAccumCount));
                _frameAccumCount += take;
                pos += take;
                if (_frameAccumCount == hop)
                {
                    RunFrame(_frameAccum, outLen);
                    outLen += hop;
                    _frameAccumCount = 0;
                }
            }

            // 整帧处理。
            while (samples.Length - pos >= hop)
            {
                RunFrame(samples.Slice(pos, hop), outLen);
                outLen += hop;
                pos += hop;
            }

            // 剩余不足一帧。
            int rem = samples.Length - pos;
            if (rem > 0)
            {
                samples.Slice(pos, rem).CopyTo(_frameAccum.AsSpan(_frameAccumCount));
                _frameAccumCount += rem;
            }

            return outLen;
        }

        private void RunFrame(ReadOnlySpan<float> hopSamples, int outOffset)
        {
            _processor!.ProcessFrame(hopSamples, _frameOut);
            _frameOut.AsSpan(0, _hopSize).CopyTo(_denoised16k.AsSpan(outOffset, _hopSize));
        }

        private void ResetState()
        {
            _processor?.Reset();
            _frameAccumCount = 0;
            _pendingHead = 0;
            _pendingCount = 0;
            _delaySkipRemaining = _delaySamplesModelRate;
        }

        public void Reset()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                ResetState();
            }
        }

        private static void EnsureCapacity(ref float[] buf, int needed)
        {
            if (needed <= buf.Length) return;
            int newSize = buf.Length == 0 ? Math.Max(needed, 1024) : buf.Length * 2;
            while (newSize < needed) newSize *= 2;
            buf = new float[newSize];
        }

        private void AppendPending(ReadOnlySpan<float> data)
        {
            if (data.IsEmpty) return;

            int required = _pendingCount + data.Length;
            int writeTail = _pendingHead + _pendingCount;

            if (writeTail + data.Length > _pendingOutput.Length)
            {
                if (required <= _pendingOutput.Length)
                {
                    if (_pendingCount > 0)
                        Array.Copy(_pendingOutput, _pendingHead, _pendingOutput, 0, _pendingCount);
                    _pendingHead = 0;
                    writeTail = _pendingCount;
                }
                else
                {
                    int newSize = _pendingOutput.Length == 0 ? Math.Max(required, 1024) : _pendingOutput.Length * 2;
                    while (newSize < required) newSize *= 2;
                    var nb = new float[newSize];
                    if (_pendingCount > 0)
                        Array.Copy(_pendingOutput, _pendingHead, nb, 0, _pendingCount);
                    _pendingOutput = nb;
                    _pendingHead = 0;
                    writeTail = _pendingCount;
                }
            }

            data.CopyTo(_pendingOutput.AsSpan(writeTail));
            _pendingCount += data.Length;
        }

        private void DrainPending(Span<float> output, int count)
        {
            _pendingOutput.AsSpan(_pendingHead, count).CopyTo(output);
            _pendingHead += count;
            _pendingCount -= count;
            if (_pendingCount == 0) _pendingHead = 0;
        }

        // ---- config.ini 解析（扁平 key=value，忽略 section；与 libDF/deepfilter-rt 一致）----
        private static Dictionary<string, string> ParseIni(string path)
        {
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            if (!File.Exists(path)) return map;
            foreach (string raw in File.ReadLines(path))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line[0] == '[' || line[0] == '#' || line[0] == ';') continue;
                int eq = line.IndexOf('=');
                if (eq <= 0) continue;
                string key = line[..eq].Trim();
                string val = line[(eq + 1)..].Trim();
                map[key] = val;
            }
            return map;
        }

        private static int GetInt(Dictionary<string, string> cfg, string key, int fallback)
            => cfg.TryGetValue(key, out var v) && int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i) ? i : fallback;

        private static float GetFloat(Dictionary<string, string> cfg, string key, float fallback)
            => cfg.TryGetValue(key, out var v) && float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var f) ? f : fallback;

        private static int GetCsvFirstInt(Dictionary<string, string> cfg, string key, int fallback)
        {
            if (!cfg.TryGetValue(key, out var v) || string.IsNullOrWhiteSpace(v)) return fallback;
            string first = v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) is { Length: > 0 } parts ? parts[0] : v;
            return int.TryParse(first, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i) ? i : fallback;
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _processor?.Dispose();
                _processor = null;
                _denoised16k = [];
                _pendingOutput = [];
                _pendingHead = 0;
                _pendingCount = 0;
                _frameAccumCount = 0;
                _disposed = true;
            }
        }
    }
}
