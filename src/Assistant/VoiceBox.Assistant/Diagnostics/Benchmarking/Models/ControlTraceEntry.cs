using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Kernel.Signals;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Diagnostics.Benchmarking.Models;

internal readonly record struct ControlTraceEntry(
    long Sequence,
    string Stage,
    long CapturePosition,
    long TurnId,
    long OperationId,
    string? Detail);
