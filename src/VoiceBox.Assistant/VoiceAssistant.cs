using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.Playback;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Wiring;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Runtime;

namespace VoiceBox.Assistant
{
    /// <summary>
    /// 完全本地化的语音对话助手。把 Mic → VAD → ASR → LLM → TTS → 播放 串成一条状态机。
    /// 可选支持 barge-in: 在 AEC/DNR 音频链路下, TTS 播放期间仍可监听用户输入并打断当前回复。
    /// 单实例只能 <see cref="StartAsync"/> 一次, 之后通过事件订阅观察对话过程。
    /// </summary>
    public sealed class VoiceAssistant : IDisposable
    {
        private readonly AssistantRuntime _runtime;

        internal VoiceAssistant(
            AssistantComposition composition,
            IOptions<VoiceAssistantOptions> options,
            ILogger<VoiceAssistant>? logger = null)
        {
            ArgumentNullException.ThrowIfNull(composition);
            _runtime = AssistantRuntimeFactory.Create(
                composition,
                options,
                logger,
                eventSender: this);
        }

        internal VoiceAssistant(
            AssistantComposition composition,
            IOptions<VoiceAssistantOptions> options,
            TimeProvider timeProvider,
            VoiceAssistantBenchmarkProbe benchmarkProbe,
            ILogger<VoiceAssistant>? logger = null,
            IPlaybackBargeInDetectorFactory? playbackBargeInDetectorFactory = null,
            Modules.Speaker.Contracts.ISpeakerModuleFactory? speakerModuleFactory = null)
        {
            ArgumentNullException.ThrowIfNull(composition);
            ArgumentNullException.ThrowIfNull(timeProvider);
            ArgumentNullException.ThrowIfNull(benchmarkProbe);
            _runtime = AssistantRuntimeFactory.Create(
                composition,
                options,
                logger,
                eventSender: this,
                timeProvider,
                benchmarkProbe,
                playbackBargeInDetectorFactory,
                speakerModuleFactory);
        }

        /// <summary>Create a typed, immutable assistant composition.</summary>
        public static VoiceAssistantBuilder CreateBuilder() => new();

        public AssistantState State => _runtime.State;

        public AssistantWakeMode WakeMode => _runtime.WakeMode;

        /// <summary>
        /// Enqueues text for speech using the assistant's global TTS configuration. The returned
        /// completion task finishes after the request has been fully played or cancelled. Queue
        /// placement affects only requests that are waiting and never interrupts current playback.
        /// </summary>
        public ValueTask<SpeechEnqueueResult> EnqueueSpeechAsync(
            string text,
            SpeechPlaybackQueuePosition queuePosition = SpeechPlaybackQueuePosition.Tail,
            CancellationToken cancellationToken = default)
            => _runtime.EnqueueSpeechAsync(text, queuePosition, cancellationToken);

        /// <summary>Compatibility overload that appends to the queue.</summary>
        public ValueTask<SpeechEnqueueResult> EnqueueSpeechAsync(
            string text,
            CancellationToken cancellationToken)
            => EnqueueSpeechAsync(text, SpeechPlaybackQueuePosition.Tail, cancellationToken);

        /// <summary>
        /// Stops only the speech request currently playing. Pending requests remain queued and
        /// playback continues with the next request. Does nothing when no request is active.
        /// </summary>
        public void StopCurrentSpeechPlayback()
            => _runtime.StopCurrentSpeechPlayback();

        public event EventHandler<AssistantStateChangedEventArgs>? StateChanged
        {
            add => _runtime.StateChanged += value;
            remove => _runtime.StateChanged -= value;
        }

        public event EventHandler<AssistantWakeModeChangedEventArgs>? WakeModeChanged
        {
            add => _runtime.WakeModeChanged += value;
            remove => _runtime.WakeModeChanged -= value;
        }

        public event EventHandler<AssistantWakeupDetectedEventArgs>? WakeupDetected
        {
            add => _runtime.WakeupDetected += value;
            remove => _runtime.WakeupDetected -= value;
        }

        public event EventHandler<UserSpokeEventArgs>? UserSpoke
        {
            add => _runtime.UserSpoke += value;
            remove => _runtime.UserSpoke -= value;
        }

        public event EventHandler<BargeInValidationEventArgs>? BargeInValidation
        {
            add => _runtime.BargeInValidation += value;
            remove => _runtime.BargeInValidation -= value;
        }

        public event EventHandler<AssistantFaultedEventArgs>? Faulted
        {
            add => _runtime.Faulted += value;
            remove => _runtime.Faulted -= value;
        }

        public event EventHandler<AssistantTextEventArgs>? AssistantText
        {
            add => _runtime.AssistantText += value;
            remove => _runtime.AssistantText -= value;
        }

        public event EventHandler<AssistantResponseGeneratedEventArgs>? AssistantResponseGenerated
        {
            add => _runtime.AssistantResponseGenerated += value;
            remove => _runtime.AssistantResponseGenerated -= value;
        }

        public event EventHandler<AssistantReplyEventArgs>? AssistantReplied
        {
            add => _runtime.AssistantReplied += value;
            remove => _runtime.AssistantReplied -= value;
        }

        public Task StartAsync(CancellationToken ct = default)
            => _runtime.StartAsync(ct);

        public Task StopAsync()
            => _runtime.StopAsync();

        public void Dispose()
            => _runtime.Dispose();

    }
}
