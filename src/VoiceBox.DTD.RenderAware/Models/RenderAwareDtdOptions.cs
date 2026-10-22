using VoiceBox.Core.Onnx;

namespace VoiceBox.DTD.RenderAware.Models;

/// <summary>Configuration for the render-aware streaming double-talk detector.</summary>
public sealed class RenderAwareDtdOptions : OnnxExecutionOptions
{
    public const uint RequiredSampleRate = 16000;
    public const uint RequiredChannelCount = 1;
    public const float ModelValidationThreshold = 0.55f;

    private string? _modelPath;

    /// <summary>
    /// ONNX model path. When null or blank, the model packaged under
    /// runtimes/models/dtd/render-aware is used.
    /// </summary>
    public string? ModelPath
    {
        get => _modelPath;
        set => _modelPath = string.IsNullOrWhiteSpace(value) ? null : value;
    }

    /// <summary>
    /// Threshold applied to the smoothed probability. The default is the model validation-set
    /// F1 operating point and must be calibrated for the deployment environment.
    /// </summary>
    public float Threshold { get; set; } = ModelValidationThreshold;

    /// <summary>
    /// ONNX intra-op thread count. This small 20 ms streaming model is fastest and most predictable
    /// with one thread by default.
    /// </summary>
    public override int ThreadsNum { get; set; } = 1;

    internal string ResolveModelPath()
        => Path.GetFullPath(ModelPath ?? Path.Combine(
            AppContext.BaseDirectory,
            "runtimes",
            "models",
            "dtd",
            "render-aware",
            "render_aware_dtd.onnx"));

    internal void Validate()
    {
        if (!float.IsFinite(Threshold) ||
            Threshold < 0f ||
            Threshold > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Threshold),
                Threshold,
                "Threshold must be finite and in [0, 1].");
        }

        if (ThreadsNum < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(ThreadsNum),
                ThreadsNum,
                "ThreadsNum must be non-negative.");
        }
    }
}
