using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.DNR.GRENet.Models;

/// <summary>
/// GRE-Net causal streaming dereverberation options.
/// </summary>
public sealed class GRENetOptions : OnnxExecutionOptions
{
    private int? _threadsNum;

    /// <summary>ONNX intra-op threads. Defaults to two on CPU and one otherwise.</summary>
    public override int ThreadsNum
    {
        get => _threadsNum ?? (ExecutionProvider == OnnxExecutionProvider.Cpu ? 2 : 1);
        set => _threadsNum = value;
    }

    /// <summary>Input sample rate. GRE-Net accepts 16 kHz audio only.</summary>
    public uint SampleRate { get; set; } = 16000;

    /// <summary>Input channel count. GRE-Net accepts mono audio only.</summary>
    public uint ChannelCount { get; set; } = 1;

    /// <summary>Input sample format. The service API always uses Float32.</summary>
    public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

    /// <summary>Path to a custom streaming ONNX model. An empty value loads the bundled stream.onnx on either provider.</summary>
    public string Model { get; set; } = string.Empty;
}
