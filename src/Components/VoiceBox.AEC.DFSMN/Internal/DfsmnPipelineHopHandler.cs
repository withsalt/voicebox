namespace VoiceBox.AEC.DFSMN.Internal
{
    internal delegate void DfsmnPipelineHopHandler(
        ReadOnlySpan<float> rawCapture,
        ReadOnlySpan<float> linearResidual,
        ReadOnlySpan<float> processedCapture,
        float vadProbability,
        in VoiceBox.Core.Services.AudioService.Models.CaptureRange captureRange);
}
