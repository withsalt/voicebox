using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Diagnostics.Benchmarking;
using VoiceBox.Assistant.Diagnostics.Metrics;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using VoiceBox.Assistant.Modules.ModelInteraction.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.Response.Controllers;
using VoiceBox.Assistant.Modules.Response.Adapters;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Realtime.Resampling;
using VoiceBox.Assistant.Realtime.Routing;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Composition.Wiring.Models;
using VoiceBox.Assistant.Modules.AudioFrontend.Adapters;
using VoiceBox.Assistant.Modules.Speaker.Controllers;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Composition.Configuration;
using VoiceBox.Assistant.Runtime.Lifecycle.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Runtime.Control.Adapters;
using VoiceBox.Assistant.Modules.Playback.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.Conversation.Stores;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Controllers;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Assistant.Runtime.Effects.Resources;
using VoiceBox.Assistant.Modules.TurnBoundary.Controllers;
using VoiceBox.Assistant.Runtime.Effects.Execution;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Runtime.Gates;
using VoiceBox.Assistant.Runtime.TurnContext;
using VoiceBox.Assistant.Runtime.Effects.Handlers;
using VoiceBox.Assistant.Modules.TurnBoundary.Adapters;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Modules.BargeIn.Adapters;
using VoiceBox.Assistant.Runtime.Pipelines;

namespace VoiceBox.Assistant.Composition.Wiring;

/// <summary>
/// Builds the Runtime Shell's collaborator graph. Every concrete instance is created here, so the
/// shell never resolves, discovers or constructs a capability. Two-phase <c>Bind</c> calls close
/// the few genuine cycles without turning any collaborator into a service bag.
/// </summary>
internal static class AssistantRuntimeCompositionFactory
{
    public static AssistantRuntimeCompositionServices Create(
        object eventSender,
        AssistantRuntimePlan runtimePlan,
        ModuleLifecyclePlan lifecycle,
        AssistantInstalledModules installed,
        IOptions<VoiceAssistantOptions> options,
        ILogger logger,
        TimeProvider timeProvider,
        VoiceAssistantBenchmarkProbe? benchmarkProbe)
    {
        ArgumentNullException.ThrowIfNull(eventSender);
        ArgumentNullException.ThrowIfNull(runtimePlan);
        ArgumentNullException.ThrowIfNull(lifecycle);
        ArgumentNullException.ThrowIfNull(installed);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(logger);
        ArgumentNullException.ThrowIfNull(timeProvider);

        VoiceAssistantOptions supplied = options.Value
            ?? throw new ArgumentException("Options value is null.", nameof(options));
        VoiceAssistantOptions snapshot = supplied.Clone();
        VoiceAssistantOptionsValidator.Validate(snapshot, nameof(options));
        VoiceAssistantBargeInOptions bargeInOptions = snapshot.BargeIn;
        VoiceAssistantListeningOptions listeningOptions = snapshot.Listening;

        IAssistantAudioFrontend audioFrontend = runtimePlan.Audio;
        IAssistantAudioOutput audioOutput = audioFrontend.Output;
        ITurnBoundarySource turnBoundary = runtimePlan.TurnBoundary;
        IWakeupModule? wakeup = installed.Wakeup;
        // Non-null exactly when Speaker.Confirmation.Enable is on. Identity gates below are
        // pass-throughs when confirmation is off, including a KWS-only composition.
        ISpeakerModule? speaker = installed.Speaker;
        IPlaybackBargeInDetector? detector = installed.BargeInDetector;
        bool bargeInInstalled = detector is not null;
        bool identityRequired = speaker is not null;
        bool wakeupInstalled = wakeup is not null;

        // --- Runtime infrastructure: no conversation semantics, no module knowledge. -----------
        var lifecycleLock = new object();
        RuntimeRunScope runScope = new();
        InputGate inputGate = new();
        OperationCorrelationGate correlation = new();
        ControlOperationRegistry operations = new();
        AssistantControlIngress ingress = new();
        // Offline evaluation may yield between virtual callback periods until every admitted
        // interruption signal has been reduced. This is a passive observation of the existing
        // admission lifecycle; it does not dispatch, acknowledge or alter any signal.
        benchmarkProbe?.AttachInterruptionAdmissionView(operations);
        TurnContextStore turnContexts = new(correlation);
        TurnSnapshotStore snapshots = new();
        TurnStateProjection projection = new();
        RuntimeConversationStateMirror stateMirror = new();
        AssistantTaskSupervisor tasks = new(logger);
        AssistantControlLoopHost loopHost = new(ingress, timeProvider);
        ControlPlaneAvailability availability = new(runScope, ingress, loopHost);

        AssistantEventDispatcher events = new(eventSender, logger);
        OrderedPublicationBoundary publications = new(
            events,
            timeProvider,
            faults: ingress,
            startWorker: false);
        BargeInDecisionJournal journal = new();

        AssistantControlSignalGateway controlSignals = new(
            ingress,
            operations,
            correlation,
            availability,
            snapshots,
            turnContexts,
            inputGate,
            runScope);

        // --- Shared value objects and playback/diagnostics collaborators. ----------------------
        AssistantAudioFormat audioFormat = new(runtimePlan.Audio.Input, runtimePlan.SpeechSynthesis);
        PlaybackStartupGuard startupGuard = new(timeProvider);
        PlaybackTextEchoGuard echoGuard = new();
        BargeInDiagnosticsCoordinator diagnostics = new(listeningOptions, audioFormat, journal, logger);
        AecMetricsSampler aecMetrics = new(runtimePlan.AcousticDiagnostics, logger, timeProvider);
        AudioResamplerPool resamplers = new(audioFormat.AudioSampleRate, audioFormat.AudioChannels);
        // Tool retention only applies when a tool runtime is installed; a text-only composition
        // keeps the message-based trim path and its byte-identical prompts.
        bool toolsInstalled = installed.ResponseGenerator is IToolHistorySource;
        ConversationHistoryStore history = new(
            snapshot.SystemPrompt,
            snapshot.MaxHistoryTurns,
            toolsInstalled ? snapshot.Tools.ToolTranscriptRetentionTurns : 0,
            snapshot.Tools.MaxPersistedToolResultBytes);

        BargeInDataPlaneController bargeInDataPlane = new(
            detector,
            journal,
            ingress,
            runScope);

        AssistantRuntimePublicEventEffectHandler publicEventEffects = new(
            publications,
            logger,
            bargeInDataPlane,
            projection,
            journal);

        // --- Turn observation owners that replace the retired observation store. ---------------
        int minimumDurationMs = bargeInOptions.MinDurationMs;
        BargeInDetector candidateDetector = new(
            minimumDurationMs,
            bargeInOptions.MinRms,
            audioFormat.AudioSampleRate,
            audioFormat.AudioChannels);
        BargeInConfirmationSettings thinkingSettings = new(minimumDurationMs, bargeInOptions.MinRms);
        BargeInConfirmationSettings speakingSettings = new(
            minimumDurationMs,
            bargeInOptions.MinRms,
            bargeInOptions.PlaybackCandidateSuppressionMs,
            recognitionSuppressionMs: 0,
            maxCandidateMs: bargeInOptions.PlaybackMaxCandidateMs,
            recognitionAudioLimitMs: bargeInOptions.PlaybackRecognitionAudioLimitMs);
        BargeInConfirmationPolicy candidatePolicy = new(
            thinkingSettings,
            speakingSettings,
            startupGuard);
        BargeInConfirmationSettings? listeningGate =
            listeningOptions.MinDurationMs > 0 || listeningOptions.MinRms > 0
                ? new BargeInConfirmationSettings(listeningOptions.MinDurationMs, listeningOptions.MinRms)
                : null;

        SpeechCandidateOwner candidates = new(
            candidateDetector,
            candidatePolicy,
            listeningGate,
            snapshots,
            turnContexts);
        BargeInValidationTurnOwner validation = new(snapshots);
        RecognitionModule recognition = new(
            installed.Recognition,
            controlSignals,
            logger);
        RecognitionTurnOwner recognitionTurn = new(
            recognition,
            snapshots,
            validation,
            validation,
            candidates);
        ConfirmedBargeInApplier confirmedBargeIn = new(
            snapshots,
            candidates,
            validation,
            recognitionTurn,
            turnContexts);
        TurnObservationRelease observationRelease = new(
            snapshots,
            recognitionTurn,
            validation,
            candidates,
            turnContexts);

        // --- The single Kernel commit port. Effects are bound once their handlers exist. -------
        AssistantControlLoopTurnWriter turnWriter = new(
            snapshots,
            projection,
            [stateMirror],
            publicEventEffects,
            loopHost,
            benchmarkProbe);

        // --- Capability modules. ----------------------------------------------------------------
        ActiveNearEndSignalSink nearEndSignals = new(ingress, operations, benchmarkProbe);
        PlaybackStartedIngressSink playbackStartedIngress = new(ingress);
        ResponseTurnResources responseResources = new(
            turnContexts,
            turnContexts,
            snapshots,
            operations,
            availability,
            runScope);
        responseResources.Bind(controlSignals);

        PlaybackModule playbackSessions = new(
            runtimePlan.SpeechSynthesis,
            audioOutput,
            logger,
            snapshot,
            resamplers,
            playbackStartedIngress,
            echoGuard);
        if (installed.ResponseGenerator is IToolHistorySource toolHistory)
            toolHistory.BindToolHistorySink(history);
        PlaybackInterruptionController playbackInterruption = new(
            audioOutput,
            startupGuard,
            playbackSessions,
            logger);
        ResponseModule responses = new(
            snapshot,
            bargeInInstalled,
            installed.ResponseGenerator,
            playbackSessions,
            history,
            controlSignals,
            publications,
            startupGuard,
            logger,
            bargeInDataPlane,
            echoGuard);

        if (installed.ResponseGenerator is IBackgroundToolResultSource backgroundResults)
        {
            backgroundResults.BindBackgroundResultSink(new PlaybackBackgroundToolResultSink(
                new BackgroundResponsePlayback(snapshots, controlSignals, responses, responseResources),
                publications,
                history));
        }

        BargeInTurnCoordinator bargeInTurns = new(
            snapshots,
            candidates,
            confirmedBargeIn,
            validation,
            recognitionTurn,
            diagnostics,
            journal,
            playbackInterruption,
            history,
            turnContexts,
            turnContexts,
            controlSignals,
            runScope,
            inputGate,
            wakeup,
            snapshot,
            identityRequired,
            logger,
            benchmarkProbe);

        object nearEndOwner = detector is null
            ? NullActiveNearEnd.Instance
            : new ActiveNearEndCoordinator(
                bargeInOptions,
                recognition,
                bargeInTurns,
                detector,
                audioFormat,
                logger,
                runScope,
                playbackInterruption,
                nearEndSignals,
                speaker,
                timeProvider);
        INearEndCapturePort nearEndCapture = (INearEndCapturePort)nearEndOwner;
        INearEndRenderPort nearEndRender = (INearEndRenderPort)nearEndOwner;
        INearEndEndpointPort nearEndEndpoint = (INearEndEndpointPort)nearEndOwner;
        INearEndConfirmationStore nearEndConfirmations = (INearEndConfirmationStore)nearEndOwner;
        // The VAD speaker gate locates every candidate chunk on the processed source timeline. With
        // barge-in installed the near-end coordinator's reservoir is that timeline; a KWS-enabled
        // composition that verifies single talk without barge-in retains its own processed reservoir.
        bool singleTalkEffective = speaker is not null &&
            speaker.Policy.Get(SpeakerVerificationPurpose.SingleTalk).Enabled;
        ProcessedSpeechTimelineLocator? standaloneSpeechTimeline = detector is null && singleTalkEffective
            ? new ProcessedSpeechTimelineLocator(
                audioFormat.SamplesFromMillisecondsToInt32(
                    snapshot.Speaker.AudioRetentionMs,
                    audioFormat.AudioSampleRate,
                    audioFormat.AudioChannels))
            : null;
        ISpeechTimelineLocator speechTimeline = standaloneSpeechTimeline ?? (ISpeechTimelineLocator)nearEndOwner;
        INearEndCaptureControlPort nearEndCaptureControl =
            (INearEndCaptureControlPort)nearEndOwner;
        INearEndResetPort nearEndReset = (INearEndResetPort)nearEndOwner;

        RecognitionTurnCoordinator recognitionTurns = new(
            recognition,
            recognitionTurn,
            snapshots,
            turnContexts,
            operations,
            bargeInTurns,
            nearEndCaptureControl,
            tasks,
            runScope,
            inputGate,
            logger);
        bargeInTurns.BindRecognition(recognitionTurns);

        VadSpeakerGateHost speakerGateHost = new(ingress, operations, tasks, runScope);
        VadSpeakerGateCoordinator speakerGate = new(
            speaker,
            speakerGateHost,
            speechTimeline,
            logger);

        SpeechBoundaryIngress speechIngress = new(
            ingress,
            projection,
            snapshots,
            recognition,
            listeningOptions,
            nearEndEndpoint,
            speakerGate);
        turnWriter.AddCommittedObserver(speechIngress);

        SpeechBoundaryCoordinator speechBoundary = new(
            snapshots,
            projection,
            candidates,
            recognitionTurn,
            validation,
            recognition,
            speakerGate,
            diagnostics,
            startupGuard,
            wakeup,
            inputGate,
            audioFormat,
            listeningOptions,
            ingress,
            nearEndEndpoint,
            nearEndCaptureControl,
            bargeInInstalled,
            identityRequired,
            logger);
        speakerGateHost.Bind(speechBoundary);

        TurnCompletionCoordinator turnCompletion = new(
            observationRelease,
            startupGuard,
            controlSignals,
            wakeup,
            snapshot.Conversation,
            turnContexts,
            logger);

        ActiveNearEndControlHandler nearEndHandler = new(
            ingress,
            nearEndConfirmations,
            bargeInTurns,
            recognition,
            speechBoundary,
            benchmarkProbe);
        PlaybackStartedControlHandler playbackStartedHandler = new(
            snapshots,
            bargeInDataPlane,
            turnBoundary,
            nearEndReset,
            startupGuard,
            bargeInOptions.PlaybackStartupGuardMs);

        BargeInRecognitionOutcomeCoordinator? recognitionOutcome = detector is null
            ? null
            : new BargeInRecognitionOutcomeCoordinator(
                recognition,
                controlSignals,
                logger,
                echoGuard,
                benchmarkProbe);
        bargeInTurns.Bind(turnCompletion);
        recognitionTurns.Bind(recognitionOutcome);

        // --- Closed effect dispatch. -------------------------------------------------------------
        AssistantEffectRunner effectRunner = new(
            new AssistantEffectHandlers(
                new InputGateEffectHandler(inputGate),
                new RecognitionEffectHandler(
                    recognition,
                    turnContexts,
                    controlSignals,
                    tasks,
                    operations,
                    runScope),
                new ResponseEffectHandler(
                    responses,
                    turnContexts,
                    responseResources,
                    tasks,
                    logger,
                    snapshot.Conversation.ContinuousConversationEnabled,
                    wakeup,
                    benchmarkProbe),
                new PlaybackEffectHandler(
                    playbackInterruption,
                    benchmarkProbe),
                new ConversationEffectHandler(history),
                publicEventEffects,
                new RuntimeOperationEffectHandler(responses, turnCompletion)));
        turnWriter.BindEffects(effectRunner);
        turnCompletion.BindEffects(effectRunner);

        // --- Data-plane routing and lifecycle. ---------------------------------------------------
        // The speaker module never consumes the final processed capture: identity frames are cut
        // from the duplex frame's after-processor timeline and routed before the DTD sees the frame.
        List<IProcessedCaptureFrameSink> captureSinks = new(2);
        // The standalone timeline must retain a frame before the VAD (inside the conversation
        // pipeline) can synchronously emit a chunk cut from it.
        if (standaloneSpeechTimeline is not null)
            captureSinks.Add(standaloneSpeechTimeline);
        List<ISpeakerCaptureFrameSink> speakerSinks = new(1);
        SpeakerCaptureFrameFactory? speakerFrames = null;
        if (speaker is not null)
        {
            speakerSinks.Add(new SpeakerCapturePipeline(speaker, playbackInterruption));
            speakerFrames = new SpeakerCaptureFrameFactory(speaker.Descriptor);
        }

        WakeupInterruptOwner wakeupInterrupts = new(
            ingress,
            snapshots,
            wakeup,
            turnBoundary,
            turnCompletion,
            bargeInDataPlane,
            nearEndReset,
            history,
            playbackInterruption,
            recognitionTurns,
            tasks,
            inputGate,
            logger);

        captureSinks.Add(new CaptureConversationPipeline(
            wakeup,
            inputGate,
            turnBoundary,
            aecMetrics,
            bargeInTurns,
            runScope,
            startupGuard,
            wakeupInterrupts,
            speechIngress,
            nearEndCapture,
            logger));

        IRenderReferenceFrameSink[] renderSinks = detector is { } renderDetector
            ? [new RenderReferencePipeline(
                renderDetector,
                aecMetrics,
                startupGuard,
                bargeInOptions,
                timeProvider,
                bargeInTurns,
                nearEndRender,
                benchmarkProbe,
                logger,
                runScope,
                playbackInterruption,
                bargeInDataPlane)]
            : [];
        AudioFrameRouter router = new(captureSinks, renderSinks, speakerSinks, speakerFrames);

        AssistantRuntimeSubscriptionOwner subscriptions = new(
            audioFrontend.ProcessedCapture,
            router.HasDuplexSinks ? audioFrontend.DuplexFrames : null,
            turnBoundary,
            router.RouteProcessedCapture,
            router.RouteDuplexFrame,
            speechIngress.OnSpeechStarted,
            speechIngress.OnSpeaking,
            speechIngress.OnSpeechEnded);

        AssistantRuntimeLifecycleCoordinator moduleLifecycle = new(
            lifecycle,
            runtimePlan.Ownership,
            audioFrontend,
            wakeup,
            speaker,
            detector,
            tasks,
            logger,
            installed.ResponseGenerator);

        AssistantRuntimeLifecycle runtimeLifecycle = new(
            lifecycleLock,
            runScope,
            moduleLifecycle,
            stateMirror,
            loopHost,
            availability,
            controlSignals,
            operations,
            ingress,
            subscriptions,
            tasks,
            turnCompletion,
            recognitionTurns,
            speakerGate,
            bargeInDataPlane,
            nearEndReset,
            playbackInterruption,
            publications,
            turnContexts,
            resamplers,
            responses,
            playbackSessions,
            inputGate,
            wakeupInstalled,
            logger);

        AssistantFaultConverger faults = new(
            moduleLifecycle.StateMachine,
            lifecycleLock,
            operations,
            bargeInDataPlane,
            tasks,
            runScope,
            inputGate,
            logger);
        faults.Bind(runtimeLifecycle);

        AssistantControlLoopRouter router2 = new(
            loopHost,
            operations,
            snapshots,
            turnWriter,
            projection,
            turnContexts,
            speechIngress,
            speechBoundary,
            recognitionTurns,
            bargeInTurns,
            nearEndHandler,
            nearEndCaptureControl,
            playbackStartedHandler,
            wakeupInterrupts,
            speakerGateHost,
            faults,
            tasks,
            runScope);
        loopHost.Bind(router2);

        AssistantRuntimeHostSurface publicSurface = new(
            runtimeLifecycle,
            events,
            snapshots,
            speechIngress,
            turnContexts,
            recognitionTurns,
            wakeup);

        var services = new AssistantRuntimeCompositionServices
        {
            Lifecycle = runtimeLifecycle,
            HostSurface = publicSurface,
            SpeechPlayback = playbackSessions,
        };
        // A failed constructor/bind must not leave a thread rooting the partial graph and sender.
        publications.Start();
        return services;
    }
}
