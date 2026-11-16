using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;

internal readonly record struct DtdProcessBatch(int HopCount, int ObservationCount, bool HasObservation, float Probability, float SmoothedProbability, bool IsNearEndSpeech, bool StateReset, RenderTimelineStatus RenderStatus);
