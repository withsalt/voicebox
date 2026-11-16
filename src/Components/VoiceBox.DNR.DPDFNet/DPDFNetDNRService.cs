using Microsoft.Extensions.Options;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DNR.DPDFNet.Internal;
using VoiceBox.DNR.DPDFNet.Models;

namespace VoiceBox.DNR.DPDFNet
{
    /// <summary>
    /// 基于 DPDFNet 的实时数字降噪服务。
    ///
    /// 工作管线：
    ///   input(16 kHz, mono, f32)
    ///     → 按 hop=160 (10ms) 分帧：滑窗 STFT → ONNX 单帧推理 (循环状态) → ISTFT 重叠相加
    ///     → 写回 buffer (剩余样本保留至下次调用)。
    /// 固有延迟约一个 hop (~10ms STFT/ISTFT)。与 <see cref="IDigitalNoiseReductionService"/>
    /// 一致，单次调用最多回写 <c>buffer.Length</c> 个样本，因管线延迟未就绪的样本会缓存到下次调用。
    /// </summary>
    public sealed class DPDFNetDNRService : IDPDFNetDNRService
    {
        private const int ModelRate = 16000;

        private readonly DPDFNetOptions _options;
        private readonly int _hopSize;
        private readonly int _freqBins;
        private readonly float? _attenAlpha;
        private readonly DPDFNetAttenuationMode _attenMode;

        private DpdfnetOnnxRunner? _runner;
        private readonly StftStreamingAnalyzer _analyzer;
        private readonly IstftStreamingSynthesizer _synthesizer;

        // 谱缓冲 (复用)。
        private readonly float[] _inRe;
        private readonly float[] _inIm;
        private readonly float[] _outRe;
        private readonly float[] _outIm;

        // 16k 域：不足一帧的累积尾部、16k 降噪输出累积。
        private readonly float[] _frameAccum;   // 长度 hop，累积到满即处理
        private int _frameAccumCount;
        private float[] _denoised16k = [];

        // 16k 输出域的待输出环形缓冲。
        private float[] _pendingOutput = [];
        private int _pendingHead;
        private int _pendingCount;
        private readonly int _delaySamples;
        private int _delaySkipRemaining;

        private readonly Lock _lock = new();
        private bool _disposed;

        /// <summary>
        /// 输入采样率下一帧 (10 ms) 的样本数，供调用方设置缓冲尺寸参考。实际 Process 接受任意长度。
        /// </summary>
        public int FrameSize { get; }
        public int DelaySamples { get; }

        public DPDFNetDNRService(IOptions<DPDFNetOptions> options)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));

            if (_options.ChannelCount != 1)
                throw new NotSupportedException($"DPDFNetDNRService only supports mono audio (ChannelCount=1), got {_options.ChannelCount}.");
            if (_options.SampleFormat != SampleFormat.Float32)
                throw new NotSupportedException($"DPDFNetDNRService only supports Float32, got {_options.SampleFormat}.");
            if (_options.SampleRate != ModelRate)
                throw new NotSupportedException($"DPDFNetDNRService only supports {ModelRate} Hz audio, got {_options.SampleRate} Hz.");

            string modelPath = ResolveModelPath(_options);
            _runner = new DpdfnetOnnxRunner(modelPath, _options);

            if (_runner.SampleRate != ModelRate)
                throw new InvalidOperationException($"Model sample rate {_runner.SampleRate} != {ModelRate}.");

            _hopSize = _runner.HopSize;
            _freqBins = _runner.FreqBins;
            int fftSize = _runner.FftSize;
            _delaySamples = Math.Max(0, fftSize - _hopSize);

            if (!string.Equals(_runner.WindowType, "vorbis", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException($"Unsupported STFT window '{_runner.WindowType}'; only 'vorbis' is implemented.");

            float[] window = VorbisWindow.Create(fftSize);
            var dft = new RealDft(fftSize);
            _analyzer = new StftStreamingAnalyzer(fftSize, _hopSize, window, dft);
            _synthesizer = new IstftStreamingSynthesizer(fftSize, _hopSize, window, dft);

            _attenAlpha = ComputeAttenuationAlpha(_options.AttenuationLimitDb);
            _attenMode = _options.AttenuationMode;

            _inRe = new float[_freqBins];
            _inIm = new float[_freqBins];
            _outRe = new float[_freqBins];
            _outIm = new float[_freqBins];
            _frameAccum = new float[_hopSize];

            FrameSize = _hopSize;
            DelaySamples = _delaySamples;
            _delaySkipRemaining = _delaySamples;
        }

        private static float? ComputeAttenuationAlpha(float? attenLimitDb)
        {
            if (attenLimitDb is null) return null;
            float v = attenLimitDb.Value;
            if (float.IsNaN(v) || v < 0f)
                throw new ArgumentException("AttenuationLimitDb must be non-negative or null.", nameof(attenLimitDb));
            if (float.IsPositiveInfinity(v)) return 0f; // 完全采用增强谱。
            return (float)Math.Pow(10.0, -v / 20.0);
        }

        private static string ResolveModelPath(DPDFNetOptions options)
        {
            if (!string.IsNullOrWhiteSpace(options.ModelPath))
            {
                if (!File.Exists(options.ModelPath))
                    throw new FileNotFoundException($"DPDFNet model file not found: {options.ModelPath}", options.ModelPath);
                return options.ModelPath;
            }

            string path = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "dnr", "dpdfnet", options.Model.ToFileName());
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"Built-in DPDFNet model '{options.Model}' not found at '{path}'. " +
                    "Ensure the model files are copied to the output directory, or set DPDFNetOptions.ModelPath.", path);
            return path;
        }

        public int Process(Span<float> buffer)
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_runner is null)
                    throw new InvalidOperationException("DPDFNet runner is not initialized.");

                int inputLen = buffer.Length;

                // 1) 16k 流式分帧降噪 → _denoised16k。
                int produced16k = ProcessFrames(buffer);

                int outputOffset = 0;
                if (_delaySkipRemaining > 0 && produced16k > 0)
                {
                    int skipped = Math.Min(_delaySkipRemaining, produced16k);
                    outputOffset = skipped;
                    produced16k -= skipped;
                    _delaySkipRemaining -= skipped;
                }

                // 2) 16k 降噪输出 → pending。
                if (produced16k > 0)
                    AppendPending(_denoised16k.AsSpan(outputOffset, produced16k));

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
                    RunFrame(_frameAccum, _denoised16k.AsSpan(outLen, hop));
                    outLen += hop;
                    _frameAccumCount = 0;
                }
            }

            // 整帧处理。
            while (samples.Length - pos >= hop)
            {
                RunFrame(samples.Slice(pos, hop), _denoised16k.AsSpan(outLen, hop));
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

        /// <summary>处理一帧 (hop 长度) 16k 样本：STFT → ONNX → (可选衰减限制) → ISTFT。</summary>
        private void RunFrame(ReadOnlySpan<float> hopSamples, Span<float> output)
        {
            _analyzer.Push(hopSamples, _inRe, _inIm);
            _runner!.Run(_inRe, _inIm, _outRe, _outIm);

            if (_attenMode == DPDFNetAttenuationMode.MagnitudeFloor)
            {
                // 幅度域增益下限: g=|enhanced|/|noisy| 钳到 [floor, +∞), 作用回 noisy 谱并保留 noisy 相位。
                // 不混合两个相位不同的复数谱, 因此没有相位干涉伪影。floor = alpha (null => 0)。
                float floor = _attenAlpha ?? 0f;
                for (int f = 0; f < _freqBins; f++)
                {
                    float inMag = MathF.Sqrt(_inRe[f] * _inRe[f] + _inIm[f] * _inIm[f]);
                    float outMag = MathF.Sqrt(_outRe[f] * _outRe[f] + _outIm[f] * _outIm[f]);
                    float g = inMag > 1e-12f ? outMag / inMag : 0f;
                    // 抑制-only: 钳到 [floor, 1]。上限 1 很关键——否则低能频点 (|noisy|≈0) 的增益会爆炸,
                    // 把 noisy 相位的噪声放大成伪影 (实测无上限时 ASR 反而更差)。
                    if (g > 1f) g = 1f;
                    if (g < floor) g = floor;
                    _outRe[f] = _inRe[f] * g;
                    _outIm[f] = _inIm[f] * g;
                }
            }
            else if (_attenAlpha is float alpha)
            {
                // ComplexBlend (默认): result = alpha·noisy + (1-alpha)·enhanced，限制最大抑制量。
                float beta = 1f - alpha;
                for (int f = 0; f < _freqBins; f++)
                {
                    _outRe[f] = alpha * _inRe[f] + beta * _outRe[f];
                    _outIm[f] = alpha * _inIm[f] + beta * _outIm[f];
                }
            }

            _synthesizer.Push(_outRe, _outIm, output);
        }

        private void ResetState()
        {
            _analyzer.Reset();
            _synthesizer.Reset();
            _runner?.Reset();
            _frameAccumCount = 0;
            _pendingHead = 0;
            _pendingCount = 0;
            _delaySkipRemaining = _delaySamples;
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

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _runner?.Dispose();
                _runner = null;
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
