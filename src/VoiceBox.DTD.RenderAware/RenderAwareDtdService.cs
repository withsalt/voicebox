using Microsoft.Extensions.Options;
using VoiceBox.DTD.RenderAware.Internal;
using VoiceBox.DTD.RenderAware.Models;

namespace VoiceBox.DTD.RenderAware;

/// <summary>
/// Production streaming service for the render-aware near-end double-talk detector.
/// Public operations are serialized so reset and disposal cannot race native inference.
/// </summary>
public sealed class RenderAwareDtdService : IRenderAwareDtdService
{
    private const int HopLengthSamples = 320;
    private const int SmoothingWindowSize = 3;

    private readonly object _sync = new();
    private readonly float _decisionThreshold;
    private readonly StreamingOverlapBuffer _microphoneWindow;
    private readonly StreamingOverlapBuffer _renderWindow;
    private readonly RenderAwareDtdOnnxRunner _runner;
    private readonly float[] _smoothingRing;
    private int _smoothingWriteIndex;
    private int _smoothingCount;
    private double _smoothingSum;
    private long _inputHopCount;
    private long _frameCount;
    private bool _disposed;

    public RenderAwareDtdService(IOptions<RenderAwareDtdOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        RenderAwareDtdOptions configured = options.Value
            ?? throw new ArgumentException("Render-aware DTD options value cannot be null.", nameof(options));
        configured.Validate();

        _decisionThreshold = configured.Threshold;
        _microphoneWindow = new StreamingOverlapBuffer(HopLengthSamples);
        _renderWindow = new StreamingOverlapBuffer(HopLengthSamples);
        _smoothingRing = new float[SmoothingWindowSize];
        // The rolling windows are the ORT input storage, avoiding a second full-window copy.
        _runner = new RenderAwareDtdOnnxRunner(
            configured.ResolveModelPath(),
            configured,
            _microphoneWindow.WindowBuffer,
            _renderWindow.WindowBuffer);
    }

    public int HopLength => HopLengthSamples;
    public int WindowLength => RenderAwareDtdOnnxRunner.WindowLengthSamples;
    public uint SampleRate => RenderAwareDtdOptions.RequiredSampleRate;
    public uint ChannelCount => RenderAwareDtdOptions.RequiredChannelCount;

    /// <summary>Recurrent-state element count of the loaded model (layers * hidden).</summary>
    internal int StateSize => _runner.StateLength;

    /// <summary>
    /// Processes one synchronized microphone/render hop. Returns <see langword="null"/> when the
    /// overlap window is not ready.
    /// </summary>
    public RenderAwareDtdObservation? Process(
        ReadOnlySpan<float> microphone,
        ReadOnlySpan<float> render,
        uint channels,
        uint sampleRate)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ValidateInput(microphone, render, channels, sampleRate);

            _inputHopCount++;
            bool microphoneReady = _microphoneWindow.Append(microphone);
            bool renderReady = _renderWindow.Append(render);
            if (!microphoneReady || !renderReady)
                return null;

            try
            {
                float probability = _runner.RunInPlace();
                float smoothed = AddSmoothedProbability(probability);
                _frameCount++;

                long hopTicks = TimeSpan.TicksPerSecond *
                    HopLengthSamples /
                    RenderAwareDtdOptions.RequiredSampleRate;
                return new RenderAwareDtdObservation(
                    probability,
                    smoothed,
                    smoothed >= _decisionThreshold,
                    _frameCount,
                    TimeSpan.FromTicks((_inputHopCount - 1) * hopTicks),
                    TimeSpan.FromTicks(hopTicks));
            }
            catch
            {
                ResetLocked();
                throw;
            }
        }
    }

    public void Reset()
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ResetLocked();
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed)
                return;

            _disposed = true;
            _runner.Dispose();
            _microphoneWindow.Reset();
            _renderWindow.Reset();
            Array.Clear(_smoothingRing);
        }
    }

    internal void CopyCurrentStateTo(Span<float> destination)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _runner.CopyCurrentStateTo(destination);
        }
    }

    private static void ValidateInput(
        ReadOnlySpan<float> microphone,
        ReadOnlySpan<float> render,
        uint channels,
        uint sampleRate)
    {
        if (channels != RenderAwareDtdOptions.RequiredChannelCount)
        {
            throw new NotSupportedException(
                $"Render-aware DTD supports mono audio only, got {channels} channels.");
        }

        if (sampleRate != RenderAwareDtdOptions.RequiredSampleRate)
        {
            throw new NotSupportedException(
                $"Render-aware DTD supports {RenderAwareDtdOptions.RequiredSampleRate} Hz only, " +
                $"got {sampleRate} Hz.");
        }

        if (microphone.Length != HopLengthSamples)
        {
            throw new ArgumentException(
                $"Microphone hop length must be {HopLengthSamples}.",
                nameof(microphone));
        }

        if (render.Length != HopLengthSamples)
        {
            throw new ArgumentException(
                $"Render hop length must be {HopLengthSamples}.",
                nameof(render));
        }

        EnsureFinite(microphone, nameof(microphone));
        EnsureFinite(render, nameof(render));
    }

    private static void EnsureFinite(ReadOnlySpan<float> samples, string parameterName)
    {
        for (int i = 0; i < samples.Length; i++)
        {
            if (!float.IsFinite(samples[i]))
            {
                throw new ArgumentException(
                    $"Audio samples must be finite; {parameterName}[{i}] is {samples[i]}.",
                    parameterName);
            }
        }
    }

    private float AddSmoothedProbability(float probability)
    {
        if (_smoothingCount == _smoothingRing.Length)
            _smoothingSum -= _smoothingRing[_smoothingWriteIndex];
        else
            _smoothingCount++;

        _smoothingRing[_smoothingWriteIndex] = probability;
        _smoothingSum += probability;
        if (++_smoothingWriteIndex == _smoothingRing.Length)
            _smoothingWriteIndex = 0;
        return Math.Clamp((float)(_smoothingSum / _smoothingCount), 0f, 1f);
    }

    private void ResetLocked()
    {
        _runner.Reset();
        _microphoneWindow.Reset();
        _renderWindow.Reset();
        Array.Clear(_smoothingRing);
        _smoothingWriteIndex = 0;
        _smoothingCount = 0;
        _smoothingSum = 0d;
        _inputHopCount = 0;
        _frameCount = 0;
    }
}
