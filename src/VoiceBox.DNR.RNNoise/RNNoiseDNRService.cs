using Microsoft.Extensions.Options;
using RNNoise.NET;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Audio.Common.Resampling;

namespace VoiceBox.DNR.RNNoise
{
    /// <summary>
    /// 基于 RNNoise 的实时数字降噪服务。
    ///
    /// 工作管线（输入采样率 != 48k 时）：
    ///   input(SampleRate, mono, f32)
    ///     → 流式 Concentus/SpeexDSP 重采样 → 48 kHz
    ///     → RNNoise.Denoise (480-sample/10ms 帧)
    ///     → 流式 Concentus/SpeexDSP 重采样 → SampleRate
    ///     → 写回 buffer（剩余样本保留至下次调用）。
    ///
    /// 输入采样率为 48 kHz 时跳过重采样，零拷贝路径。
    /// 由于 RNNoise 以 10 ms 帧处理，最小固有延迟约 10 ms；非 48 kHz 路径还包含
    /// SpeexDSP 重采样器的群延迟。
    /// </summary>
    public sealed class RNNoiseDNRService : IRNNoiseDNRService
    {
        private const int RNNoiseNativeRate = 48000;
        private const int RNNoiseFrameSize = Denoiser.FrameSize;
        private const int RNNoiseAlgorithmicDelaySamples = Denoiser.AlgorithmicDelaySamples;

        private readonly RNNoiseOptions _options;
        private readonly int _inputRate;
        private readonly bool _needsResample;
        private readonly bool _bypass;

        private Denoiser? _denoiser;
        private DenoiseModel? _customModel;
        private readonly StreamingSpeexResampler? _upsampler;
        private readonly StreamingSpeexResampler? _downsampler;

        private float[] _work48k = [];
        private float[] _resampleOutput = [];

        private float[] _pendingOutput = [];
        private int _pendingHead;
        private int _pendingCount;
        private int _delaySkipRemaining;

        private readonly Lock _lock = new();
        private bool _disposed;

        /// <summary>
        /// 输入采样率下一帧（10 ms）的样本数。仅作为调用方缓冲区尺寸的参考。
        /// 实际 Process 接受任意长度输入。
        /// </summary>
        public int FrameSize { get; }
        public int DelaySamples { get; }

        public RNNoiseDNRService(IOptions<RNNoiseOptions> options)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));

            if (_options.ChannelCount != 1)
                throw new NotSupportedException($"RNNoiseDNRService only supports mono audio (ChannelCount=1), got {_options.ChannelCount}.");
            if (_options.SampleFormat != SampleFormat.Float32)
                throw new NotSupportedException($"RNNoiseDNRService only supports Float32, got {_options.SampleFormat}.");
            if (_options.SampleRate == 0)
                throw new ArgumentException("SampleRate must be > 0.", nameof(options));

            _inputRate = (int)_options.SampleRate;
            _needsResample = _inputRate != RNNoiseNativeRate;
            _bypass = _options.Bypass;

            if (!string.IsNullOrWhiteSpace(_options.ModelPath))
            {
                if (!File.Exists(_options.ModelPath))
                    throw new FileNotFoundException($"RNNoise model file not found: {_options.ModelPath}", _options.ModelPath);
                _customModel = new DenoiseModel(_options.ModelPath);
                _denoiser = new Denoiser(_customModel);
            }
            else
            {
                _denoiser = new Denoiser();
            }

            if (_needsResample)
            {
                _upsampler = new StreamingSpeexResampler(_inputRate, RNNoiseNativeRate);
                _downsampler = new StreamingSpeexResampler(RNNoiseNativeRate, _inputRate);
            }

            FrameSize = (int)Math.Round(RNNoiseFrameSize * (double)_inputRate / RNNoiseNativeRate);
            if (FrameSize <= 0) FrameSize = RNNoiseFrameSize;
            DelaySamples = _bypass
                ? 0
                : (int)Math.Round(RNNoiseAlgorithmicDelaySamples * (double)_inputRate / RNNoiseNativeRate);
            _delaySkipRemaining = _bypass ? 0 : RNNoiseAlgorithmicDelaySamples;
        }

        public int Process(Span<float> buffer)
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_denoiser == null)
                    throw new InvalidOperationException("Denoiser is not initialized.");

                if (_bypass)
                    return buffer.Length;

                int inputLen = buffer.Length;

                int work48kLen;
                if (_needsResample)
                {
                    work48kLen = inputLen == 0 ? 0 : _upsampler!.Process(buffer, ref _work48k);
                }
                else
                {
                    EnsureWorkCapacity(inputLen);
                    if (inputLen > 0)
                        buffer.CopyTo(_work48k.AsSpan(0, inputLen));
                    work48kLen = inputLen;
                }

                int denoised48kLen = _denoiser.Denoise(_work48k.AsSpan(0, work48kLen), finish: false);
                int denoised48kOffset = 0;
                if (_delaySkipRemaining > 0 && denoised48kLen > 0)
                {
                    int skipped = Math.Min(_delaySkipRemaining, denoised48kLen);
                    denoised48kOffset = skipped;
                    denoised48kLen -= skipped;
                    _delaySkipRemaining -= skipped;
                }

                if (denoised48kLen > 0)
                {
                    if (_needsResample)
                    {
                        int produced = _downsampler!.Process(
                            _work48k.AsSpan(denoised48kOffset, denoised48kLen),
                            ref _resampleOutput);
                        AppendPending(_resampleOutput.AsSpan(0, produced));
                    }
                    else
                    {
                        AppendPending(_work48k.AsSpan(denoised48kOffset, denoised48kLen));
                    }
                }

                int toCopy = Math.Min(_pendingCount, inputLen);
                if (toCopy > 0)
                    DrainPending(buffer, toCopy);

                return toCopy;
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                ResetState();
            }
        }

        private void ResetState()
        {
            _denoiser?.Reset();
            _upsampler?.Reset();
            _downsampler?.Reset();
            _pendingHead = 0;
            _pendingCount = 0;
            _delaySkipRemaining = _bypass ? 0 : RNNoiseAlgorithmicDelaySamples;
        }

        private void EnsureWorkCapacity(int needed)
        {
            if (needed <= _work48k.Length) return;
            int newSize = _work48k.Length == 0 ? Math.Max(needed, RNNoiseFrameSize * 4) : _work48k.Length * 2;
            while (newSize < needed) newSize *= 2;
            Array.Resize(ref _work48k, newSize);
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
            if (_pendingCount == 0)
                _pendingHead = 0;
        }

        public void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _denoiser?.Dispose();
                _denoiser = null;
                _customModel?.Dispose();
                _customModel = null;
                _work48k = [];
                _resampleOutput = [];
                _upsampler?.Dispose();
                _downsampler?.Dispose();
                _pendingOutput = [];
                _pendingHead = 0;
                _pendingCount = 0;
                _disposed = true;
            }
        }
    }
}
