using VoiceBox.DTD.RenderAware.Models;

namespace VoiceBox.Assistant.Api.Configuration;

/// <summary>
/// Render-aware DTD integration settings. Model execution settings live in
/// <c>RenderAwareDtdOptions</c>, alongside the model implementation.
/// </summary>
public sealed class VoiceAssistantRenderAwareDtdOptions
{
    /// <summary>ONNX model and inference settings for each playback detector instance.</summary>
    public RenderAwareDtdOptions Model { get; set; } = new();

    /// <summary>Recent decision history retained by the adapter.</summary>
    public int DecisionHistoryMs { get; set; } = 2000;

    /// <summary>
    /// Blocks playback interruption for this long after a significant playback-gain or
    /// coherent acoustic-path change. AEC/DTD processing and echo-profile adaptation continue
    /// during the guard. Set to 0 to disable. Default 500ms.
    /// </summary>
    public int AcousticPathChangeGuardMs { get; set; } = 500;
}
