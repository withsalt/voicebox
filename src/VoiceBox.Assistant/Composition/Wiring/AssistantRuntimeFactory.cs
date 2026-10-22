using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Composition.Wiring.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;

namespace VoiceBox.Assistant.Composition.Wiring;

/// <summary>
/// Composition-layer factory: turns a validated plan into installed modules, then constructs the
/// Runtime shell. Module construction happens here so the shell never `new`s Speaker/Wakeup
/// adapters or the render-aware detector.
/// </summary>
internal static class AssistantRuntimeFactory
{
    public static AssistantRuntime Create(
        AssistantComposition composition,
        IOptions<VoiceAssistantOptions> options,
        ILogger? logger = null,
        object? eventSender = null,
        TimeProvider? timeProvider = null,
        VoiceAssistantBenchmarkProbe? benchmarkProbe = null,
        IPlaybackBargeInDetectorFactory? playbackBargeInDetectorFactory = null,
        ISpeakerModuleFactory? speakerModuleFactory = null)
    {
        ArgumentNullException.ThrowIfNull(composition);
        ArgumentNullException.ThrowIfNull(options);

        VoiceAssistantOptions supplied = options.Value
            ?? throw new ArgumentException("Options value is null.", nameof(options));
        VoiceAssistantOptions snapshot = supplied.Clone();
        ILogger resolvedLogger = logger ?? NullLogger<AssistantRuntime>.Instance;
        TimeProvider resolvedTime = timeProvider ?? TimeProvider.System;
        AssistantRuntimePlan plan = composition.RuntimePlan;
        var audioFormat = new AssistantAudioFormat(plan.Audio.Input, plan.SpeechSynthesis);

        // Composition needs a sender identity before the facade exists; when the host supplies
        // none, the facade replaces this placeholder before any event can be raised.
        var placeholderSender = new object();
        IPlaybackBargeInDetector? detector = null;
        ISpeakerModule? speaker = null;
        IWakeupModule? wakeup = null;
        IResponseGenerator? responseGenerator = null;
        try
        {
            IPlaybackBargeInDetectorFactory? detectorFactory =
                playbackBargeInDetectorFactory ?? plan.PlaybackBargeInDetectorFactory;
            if (detectorFactory is not null)
            {
                detector = detectorFactory.Create()
                    ?? throw new InvalidOperationException(
                        "The playback barge-in detector factory returned null.");
            }
            else if (snapshot.BargeIn.Enable)
            {
                throw new InvalidOperationException(
                    "Barge-in is enabled but the composition provided no detector factory.");
            }

            if (plan.Speaker is not null)
            {
                // The identity tap descriptor is frozen by the frontend at Build; enrollment and
                // online verification compare against the same fingerprint. A benchmark/test seam
                // may substitute the module, never remove it: the plan still requires it.
                SpeakerCaptureDescriptor identityTap = plan.Audio.SpeakerCapture
                    ?? throw new InvalidOperationException(
                        "Online speaker verification is planned but the audio frontend exposes no identity tap.");
                speaker = speakerModuleFactory is not null
                    ? speakerModuleFactory.Create(identityTap, audioFormat.AudioSampleRate, audioFormat.AudioChannels, resolvedLogger)
                    : plan.Speaker.Create(identityTap, audioFormat.AudioSampleRate, audioFormat.AudioChannels, resolvedLogger);
            }

            responseGenerator = plan.ResponseGeneration.Create();

            if (plan.Wakeup is not null)
                wakeup = plan.Wakeup.Create(speaker, resolvedTime, resolvedLogger);

            var installed = new AssistantInstalledModules(
                speaker,
                wakeup,
                detector,
                plan.Recognition,
                responseGenerator);

            AssistantRuntimeCompositionServices services = AssistantRuntimeCompositionFactory.Create(
                eventSender ?? placeholderSender,
                plan,
                composition.Lifecycle,
                installed,
                options,
                resolvedLogger,
                resolvedTime,
                benchmarkProbe);
            var runtime = new AssistantRuntime(services);
            if (eventSender is null)
                services.HostSurface.BindSender(runtime);
            return runtime;
        }
        catch
        {
            try { (responseGenerator as IDisposable)?.Dispose(); } catch { }
            try { wakeup?.Dispose(); } catch { }
            try { speaker?.Dispose(); } catch { }
            try { detector?.Dispose(); } catch { }
            try { (plan.Audio as SplitAssistantAudioFrontend)?.Dispose(); } catch { }
            throw;
        }
    }

}
