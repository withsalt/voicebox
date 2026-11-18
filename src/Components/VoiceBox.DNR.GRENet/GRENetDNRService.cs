using Microsoft.Extensions.Options;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.DNR.GRENet.Internal;
using VoiceBox.DNR.GRENet.Models;

namespace VoiceBox.DNR.GRENet;

/// <summary>
/// Causal GRE-Net dereverberation service for the shared DNR slot.
///
/// The service accepts arbitrary input chunk sizes, accumulates 200-sample hops,
/// runs one ONNX frame per hop, and returns source-backed samples through a small
/// pending queue. A single lock protects the model and all streaming state because
/// audio callbacks and lifecycle resets can arrive on different threads.
/// </summary>
public sealed class GRENetDNRService : IDigitalNoiseReductionService
{
    private const int SampleRateHz = 16000;
    private const int WindowSize = 400;
    private const int HopSize = 200;
    private const float CompressFactor = 0.3f;
    private const float MaxLinearMagnitude = 10f;

    private readonly StreamingStft _stft;
    private readonly StreamingIstft _istft;
    private readonly IGRENetOnnxRunner _runner;
    private readonly RealDft _dft;
    private readonly float[] _frameInput = new float[HopSize];
    private readonly float[] _analysisReal = new float[WindowSize / 2 + 1];
    private readonly float[] _analysisImaginary = new float[WindowSize / 2 + 1];
    private readonly float[] _analysisAmplitude = new float[WindowSize / 2 + 1];
    private readonly float[] _analysisPhase = new float[WindowSize / 2 + 1];
    private readonly float[] _enhancedAmplitude = new float[WindowSize / 2 + 1];
    private readonly float[] _enhancedPhase = new float[WindowSize / 2 + 1];
    private readonly float[] _enhancedReal = new float[WindowSize / 2 + 1];
    private readonly float[] _enhancedImaginary = new float[WindowSize / 2 + 1];
    private float[] _enhancedFrames = [];
    private float[] _pendingOutput = [];
    private int _frameInputCount;
    private int _pendingHead;
    private int _pendingCount;
    private int _delaySkipRemaining;
    private readonly Lock _lock = new();
    private bool _disposed;

    /// <summary>Preferred callback size: one 12.5 ms GRE-Net hop.</summary>
    public int FrameSize => HopSize;

    /// <summary>Startup lookahead/backlog before the first source-backed hop.</summary>
    public int DelaySamples => HopSize;

    public GRENetDNRService(IOptions<GRENetOptions> options)
    {
        GRENetOptions configuration = options?.Value ?? throw new ArgumentNullException(nameof(options));
        ValidateOptions(configuration);
        string modelPath = ResolveModelPath(configuration);
        _runner = GRENetOnnxRunnerFactory.Create(modelPath, configuration);
        if (_runner.FrequencyBins != WindowSize / 2 + 1)
        {
            throw new InvalidOperationException("GRE-Net frequency-bin count does not match the configured analysis window.");
        }

        _dft = new RealDft(WindowSize);
        _stft = new StreamingStft(WindowSize, HopSize, _dft);
        _istft = new StreamingIstft(WindowSize, HopSize, _dft);
        _delaySkipRemaining = DelaySamples;
    }

    public int Process(Span<float> buffer)
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            int inputLength = buffer.Length;
            int produced = ProcessFrames(buffer);
            int producedOffset = 0;
            if (_delaySkipRemaining > 0 && produced > 0)
            {
                int skipped = Math.Min(_delaySkipRemaining, produced);
                _delaySkipRemaining -= skipped;
                producedOffset = skipped;
                produced -= skipped;
            }

            if (produced > 0)
            {
                AppendPending(_enhancedFrames.AsSpan(producedOffset, produced));
            }

            int copied = Math.Min(inputLength, _pendingCount);
            if (copied > 0)
            {
                DrainPending(buffer, copied);
            }

            return copied;
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

    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _runner.Dispose();
            _enhancedFrames = [];
            _pendingOutput = [];
            _disposed = true;
        }
    }

    private static void ValidateOptions(GRENetOptions options)
    {
        if (options.ExecutionProvider is not OnnxExecutionProvider.Cpu and not OnnxExecutionProvider.Cuda)
        {
            throw new NotSupportedException(
                $"GRENetDNRService supports CPU and CUDA execution providers only, got {options.ExecutionProvider}.");
        }

        if (options.SampleRate != SampleRateHz)
        {
            throw new NotSupportedException($"GRENetDNRService only supports {SampleRateHz} Hz audio, got {options.SampleRate}.");
        }

        if (options.ChannelCount != 1)
        {
            throw new NotSupportedException($"GRENetDNRService only supports mono audio, got {options.ChannelCount} channels.");
        }

        if (options.SampleFormat != SampleFormat.Float32)
        {
            throw new NotSupportedException($"GRENetDNRService only supports Float32 audio, got {options.SampleFormat}.");
        }
    }

    private static string ResolveModelPath(GRENetOptions options)
    {
        string path = string.IsNullOrWhiteSpace(options.Model)
            ? Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "dnr", "grenet", "stream.onnx")
            : options.Model;
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"GRE-Net model file not found at '{path}'. Check the model path and runtime assets.", path);
        }

        return path;
    }

    private int ProcessFrames(ReadOnlySpan<float> samples)
    {
        int maxFrames = (samples.Length + _frameInputCount) / HopSize;
        EnsureCapacity(ref _enhancedFrames, Math.Max(HopSize, maxFrames * HopSize));
        int outputLength = 0;
        int offset = 0;

        if (_frameInputCount > 0)
        {
            int needed = HopSize - _frameInputCount;
            int copied = Math.Min(needed, samples.Length);
            samples[..copied].CopyTo(_frameInput.AsSpan(_frameInputCount));
            _frameInputCount += copied;
            offset += copied;
            if (_frameInputCount == HopSize)
            {
                RunFrame(_frameInput, _enhancedFrames.AsSpan(outputLength, HopSize));
                outputLength += HopSize;
                _frameInputCount = 0;
            }
        }

        while (samples.Length - offset >= HopSize)
        {
            RunFrame(samples.Slice(offset, HopSize), _enhancedFrames.AsSpan(outputLength, HopSize));
            outputLength += HopSize;
            offset += HopSize;
        }

        int remaining = samples.Length - offset;
        if (remaining > 0)
        {
            samples.Slice(offset, remaining).CopyTo(_frameInput.AsSpan(_frameInputCount));
            _frameInputCount += remaining;
        }

        return outputLength;
    }

    private void RunFrame(ReadOnlySpan<float> hop, Span<float> output)
    {
        _stft.Push(hop, _analysisReal, _analysisImaginary);
        for (int i = 0; i < _analysisAmplitude.Length; i++)
        {
            float magnitude = MathF.Sqrt(_analysisReal[i] * _analysisReal[i] + _analysisImaginary[i] * _analysisImaginary[i] + 1e-9f);
            _analysisAmplitude[i] = MathF.Pow(magnitude, CompressFactor);
            _analysisPhase[i] = MathF.Atan2(_analysisImaginary[i] + 1e-10f, _analysisReal[i] + 1e-5f);
        }

        _runner.Run(_analysisAmplitude, _analysisPhase, _enhancedAmplitude, _enhancedPhase);
        for (int i = 0; i < _enhancedAmplitude.Length; i++)
        {
            float magnitude = MathF.Pow(MathF.Max(_enhancedAmplitude[i], 1e-9f), 1f / CompressFactor);
            magnitude = MathF.Min(magnitude, MaxLinearMagnitude);
            _enhancedReal[i] = magnitude * MathF.Cos(_enhancedPhase[i]);
            _enhancedImaginary[i] = magnitude * MathF.Sin(_enhancedPhase[i]);
        }

        _istft.Push(_enhancedReal, _enhancedImaginary, output);
    }

    private void ResetState()
    {
        _stft.Reset();
        _istft.Reset();
        _runner.Reset();
        _frameInputCount = 0;
        _pendingHead = 0;
        _pendingCount = 0;
        _delaySkipRemaining = DelaySamples;
    }

    private static void EnsureCapacity(ref float[] buffer, int required)
    {
        if (required <= buffer.Length)
        {
            return;
        }

        int size = buffer.Length == 0 ? Math.Max(required, HopSize * 4) : buffer.Length * 2;
        while (size < required)
        {
            size *= 2;
        }

        Array.Resize(ref buffer, size);
    }

    private void AppendPending(ReadOnlySpan<float> samples)
    {
        if (samples.IsEmpty)
        {
            return;
        }

        int required = checked(_pendingCount + samples.Length);
        int tail = _pendingHead + _pendingCount;
        if (tail + samples.Length > _pendingOutput.Length)
        {
            if (required <= _pendingOutput.Length)
            {
                Array.Copy(_pendingOutput, _pendingHead, _pendingOutput, 0, _pendingCount);
            }
            else
            {
                int size = _pendingOutput.Length == 0 ? Math.Max(required, HopSize * 4) : _pendingOutput.Length * 2;
                while (size < required)
                {
                    size *= 2;
                }

                var resized = new float[size];
                Array.Copy(_pendingOutput, _pendingHead, resized, 0, _pendingCount);
                _pendingOutput = resized;
            }

            _pendingHead = 0;
            tail = _pendingCount;
        }

        samples.CopyTo(_pendingOutput.AsSpan(tail));
        _pendingCount += samples.Length;
    }

    private void DrainPending(Span<float> destination, int count)
    {
        _pendingOutput.AsSpan(_pendingHead, count).CopyTo(destination);
        _pendingHead += count;
        _pendingCount -= count;
        if (_pendingCount == 0)
        {
            _pendingHead = 0;
        }
    }
}
