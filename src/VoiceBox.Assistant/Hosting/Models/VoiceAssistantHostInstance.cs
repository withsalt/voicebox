using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Hosting.Diagnostics;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;

namespace VoiceBox.Assistant.Hosting.Models;

internal sealed record VoiceAssistantHostInstance(
    VoiceAssistant Assistant,
    VoiceAssistantDiagnostics Diagnostics);
