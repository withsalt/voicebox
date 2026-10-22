using Microsoft.Extensions.Logging.Abstractions;
using System.Reflection;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Adapters;
using VoiceBox.Assistant.Diagnostics;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Models;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;
using Xunit;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Diagnostics.Recording;

namespace VoiceBox.Assistant.Tests;

public sealed class RecognitionModuleContractTests
{
    [Fact]
    public void RecognitionBoundary_ExposesTypedModuleSessionOutcomeAndProviderPort()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;

        Type module = Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionModule");
        Type session = Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionSession");
        Type outcome = Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Models.RecognitionOutcome");
        Type provider = Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Contracts.IRecognitionProviderAdapter");
        Type audioPort = Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Contracts.IRecognitionAudioPort");

        Assert.True(module.IsSealed);
        Assert.True(session.IsSealed);
        Assert.True(outcome.IsValueType);
        Assert.True(provider.IsInterface);
        Assert.True(audioPort.IsInterface);
    }

    [Fact]
    public void RecognitionModule_DoesNotOwnKernelRuntimeSnapshotOrProviderSpecificState()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type module = Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionModule");
        Type kernel = Require(assembly, "VoiceBox.Assistant.Kernel.TurnKernel");
        Type snapshot = Require(assembly, "VoiceBox.Assistant.Kernel.Models.TurnSnapshot");

        Assert.Null(assembly.GetType("VoiceBox.Assistant.Kernel.TurnReducer"));

        FieldInfo[] fields = module.GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        Assert.DoesNotContain(fields, field => field.FieldType == kernel);
        Assert.DoesNotContain(fields, field => field.FieldType == snapshot);
        Assert.DoesNotContain(fields, field =>
            field.FieldType.Namespace?.StartsWith(
                "VoiceBox.Core.Services.ASRService",
                StringComparison.Ordinal) == true);
    }

    [Fact]
    public void RecognitionConsumers_UseOnlyRecognitionPortsWithoutDirectAsrOwnership()
    {
        Assert.Contains(
            typeof(ActiveNearEndCoordinator).GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            field => field.FieldType == typeof(IRecognitionAudioPort));
        Assert.Contains(
            typeof(BargeInRecognitionOutcomeCoordinator).GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            field => field.FieldType == typeof(IRecognitionCompletionPort));
        Assert.Contains(
            ShellSurface.Fields,
            field => field.FieldType == typeof(IRecognitionAudioPort));

        foreach (Type consumer in new[]
        {
            typeof(AssistantRuntime),
            typeof(ActiveNearEndCoordinator),
            typeof(BargeInRecognitionOutcomeCoordinator),
        })
        {
            Assert.DoesNotContain(
                consumer.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public),
                field => typeof(IASRService).IsAssignableFrom(field.FieldType));
        }
    }

    [Fact]
    public async Task Session_StartFeedFinishAndCleanup_AreOrderedAndExactlyOnce()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        Task prepare = module.PrepareAsync(7, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        Assert.True(module.TryFeedAudio(7, [0.1f, 0.2f]));
        Assert.True(module.TryFeedAudio(7, [0.3f]));
        provider.CompleteStart("session-7");
        await prepare;

        RecognitionOutcome first = await module.FinishAsync(
            7,
            "session-7",
            TimeSpan.FromMilliseconds(90),
            CancellationToken.None);
        RecognitionOutcome duplicate = await module.FinishAsync(
            7,
            "session-7",
            TimeSpan.FromMilliseconds(90),
            CancellationToken.None);
        await module.CleanupAsync(7, "session-7");

        Assert.Equal(RecognitionOutcomeKind.Completed, first.Kind);
        Assert.Equal(first, duplicate);
        Assert.Equal(7, first.TurnId);
        Assert.Equal(1, first.OperationId);
        Assert.Equal(" raw text ", first.RawText);
        Assert.Equal(1, provider.StartCount);
        Assert.Equal(1, provider.FinishCount);
        Assert.Equal(
            ["start", "feed:0.1,0.2", "feed:0.3", "finish"],
            provider.Trace);
        Assert.Single(control.Ready);
        Assert.Equal((7L, 1L, "session-7"), control.Ready[0]);
    }

    [Fact]
    public async Task Prepare_BootstrapAudioPrecedesCaptureFramesQueuedWhileProviderStarts()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        RecognitionModule? module = null;
        control.OnRecognitionStarted = (turnId, _, sessionId) =>
        {
            Assert.True(module!.TryAttachCaptureSession(
                turnId,
                sessionId,
                isValidationTurn: false,
                bootstrapAudio: [new[] { 0.1f, 0.2f }],
                out bool speechAlreadyEnded,
                out TimeSpan speechDuration));
            Assert.False(speechAlreadyEnded);
            Assert.Equal(TimeSpan.Zero, speechDuration);
            return ValueTask.FromResult(true);
        };
        module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        Task prepare = module.PrepareAsync(7, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        Assert.True(module.TryFeedAudio(7, [0.7f]));
        provider.CompleteStart("session-7");
        await prepare;

        RecognitionOutcome outcome = await module.FinishAsync(
            7,
            "session-7",
            TimeSpan.Zero,
            CancellationToken.None);

        Assert.Equal(RecognitionOutcomeKind.Completed, outcome.Kind);
        Assert.Equal(
            ["start", "feed:0.1,0.2", "feed:0.7", "finish"],
            provider.Trace);
    }

    [Fact]
    public async Task CaptureAttach_TakesOwnedPreRollBeforePreSessionAndLiveAudio()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        RecognitionModule? module = null;
        control.OnRecognitionStarted = (turnId, _, sessionId) =>
        {
            Assert.True(module!.TryAttachCaptureSession(
                turnId,
                sessionId,
                isValidationTurn: false,
                bootstrapAudio: [new[] { 0.1f }],
                out bool speechAlreadyEnded,
                out TimeSpan speechDuration));
            Assert.False(speechAlreadyEnded);
            Assert.Equal(TimeSpan.Zero, speechDuration);

            SpeakingDecision live = module.ObserveCaptureAudio(
                turnId,
                AssistantState.Listening,
                [0.3f],
                default(BargeInValidationCaptureSnapshot));
            Assert.Equal(SpeakingDecisionKind.FeedAsr, live.Kind);
            Assert.True(((IRecognitionAudioPort)module).TryFeedAudio(turnId, [0.3f]));
            return ValueTask.FromResult(true);
        };
        module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        module.PrepareCaptureTurn(7);

        Task prepare = module.PrepareAsync(7, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        Assert.Equal(
            SpeakingDecisionKind.None,
            module.ObserveCaptureAudio(
                7,
                AssistantState.Listening,
                [0.2f],
                default(BargeInValidationCaptureSnapshot)).Kind);
        provider.CompleteStart("session-7");
        await prepare;
        RecognitionOutcome outcome = await module.FinishAsync(
            7,
            "session-7",
            TimeSpan.Zero,
            CancellationToken.None);

        Assert.Equal(RecognitionOutcomeKind.Completed, outcome.Kind);
        Assert.Equal(
            ["start", "feed:0.1", "feed:0.2", "feed:0.3", "finish"],
            provider.Trace);
    }

    [Fact]
    public async Task ActiveAudioPort_FeedsAnAdmittedSessionWithoutAddingPumpBacklog()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        var activeAudio = (IRecognitionAudioPort)module;
        Assert.False(activeAudio.TryGetProviderSession(7, out _, out _));

        provider.CompleteStart("session-7");
        await module.PrepareAsync(7, Interlocked.Increment(ref nextOperationId), CancellationToken.None);

        Assert.True(activeAudio.TryGetProviderSession(7, out long operationId, out string? sessionId));
        Assert.Equal(nextOperationId, operationId);
        Assert.Equal("session-7", sessionId);
        Assert.False(activeAudio.TryGetProviderSession(8, out _, out _));
        Assert.True(activeAudio.TryFeedAudio(7, [0.4f, 0.5f]));

        Assert.Equal(["start", "feed:0.4,0.5"], provider.Trace);

        await module.FinishAsync(7, sessionId!, TimeSpan.Zero, CancellationToken.None);
        Assert.False(activeAudio.TryGetProviderSession(7, out _, out _));
    }

    [Fact]
    public async Task CaptureRecognizing_ObservationDoesNotConsumeSessionBeforeCommit()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("session-6");
        await module.PrepareAsync(6, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        module.PrepareCaptureTurn(6);
        Assert.True(module.TryAttachCaptureSession(
            6,
            "session-6",
            isValidationTurn: false,
            bootstrapAudio: [],
            out _,
            out _));

        Assert.True(module.CanEnterCaptureRecognizing(
            6,
            "session-6",
            AssistantState.Listening,
            out bool isValidationTurn));
        Assert.False(isValidationTurn);
        Assert.Equal("session-6", module.GetCaptureSessionId(6, 0));

        Assert.True(module.CommitEnterCaptureRecognizing(
            6,
            "session-6",
            isValidationTurn: false));
        Assert.Null(module.GetCaptureSessionId(6, 0));
    }

    [Fact]
    public async Task TimeoutAcceptedValidation_PreservesAttachedSessionUntilEndpoint()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("validation-session");
        await module.PrepareAsync(9, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        module.PrepareCaptureTurn(9);
        Assert.True(module.TryAttachCaptureSession(
            9,
            "validation-session",
            isValidationTurn: true,
            bootstrapAudio: [],
            out _,
            out _));

        var turnState = new StaticTurnStateView(new TurnSnapshot
        {
            ConversationState = AssistantState.Listening,
            TurnId = 9,
            NextTurnOrdinal = 10,
        });
        var validation = new StaticValidationTurnView(9);
        var owner = new RecognitionTurnOwner(
            module,
            turnState,
            validation,
            new EmptyValidationPreRoll(),
            new EmptyCandidatePreRoll());

        owner.ClearActiveSessionId();

        Assert.Equal("validation-session", module.GetCaptureSessionId(9, 9));
        SpeechEndedDecision endpoint = owner.ObserveSpeechEnded(TimeSpan.FromMilliseconds(480));
        Assert.Equal(SpeechEndedDecisionKind.ReadyForRecognition, endpoint.Kind);
        Assert.Equal("validation-session", endpoint.AsrSessionId);
    }

    [Fact]
    public async Task AbandonedRecognizingSession_IsRecoveredFromModuleOwnershipRegistry()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("validation-session");
        await module.PrepareAsync(9, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        module.PrepareCaptureTurn(9);
        Assert.True(module.TryAttachCaptureSession(
            9,
            "validation-session",
            isValidationTurn: true,
            bootstrapAudio: [],
            out _,
            out _));
        Assert.True(module.CommitEnterCaptureRecognizing(
            9,
            "validation-session",
            isValidationTurn: true));
        Assert.Null(module.GetCaptureSessionId(9, 9));

        Assert.Equal("validation-session", module.TakeAbandonedCaptureSession(9, 9));
        await module.CleanupAbandonedAsrSessionAsync("validation-session");
        Assert.Equal(1, provider.FinishCount);
    }

    [Fact]
    public async Task PrepareFailure_PublishesCorrelatedFailedOutcome()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 40;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        Task prepare = module.PrepareAsync(9, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        provider.FailStart(new InvalidOperationException("start failed"));
        await prepare;

        RecognitionOutcome failed = Assert.Single(control.Failed);
        Assert.Equal(RecognitionOutcomeKind.Failed, failed.Kind);
        Assert.Equal(9, failed.TurnId);
        Assert.Equal(41, failed.OperationId);
        Assert.Equal("session-start-failed", failed.Failure.MessageKey);
    }

    [Fact]
    public async Task RunCancelledWhileProviderStarts_RetiresPendingSessionWithoutPublishingReady()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        using var runCancellation = new CancellationTokenSource();
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        Task prepare = module.PrepareAsync(
            14,
            Interlocked.Increment(ref nextOperationId),
            runCancellation.Token);
        Assert.True(module.TryFeedAudio(14, [0.1f, 0.2f]));
        runCancellation.Cancel();
        provider.CompleteStart("session-disposed-start");

        await prepare.WaitAsync(TimeSpan.FromSeconds(1));

        Assert.Equal(["start"], provider.Trace);
        Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByTurn"));
        Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByProviderId"));
        Assert.Empty(control.Ready);
    }

    [Fact]
    public async Task SameIngressSchedule_ReplaysOneHundredTimesWithIdenticalProviderTrace()
    {
        string[]? expected = null;
        for (int iteration = 0; iteration < 100; iteration++)
        {
            var provider = new RecordingRecognitionProvider();
            var control = new RecordingRecognitionControlPort();
            long nextOperationId = 0;
            var module = new RecognitionModule(
                provider,
                control,
                NullLogger.Instance);

            Task prepare = module.PrepareAsync(1, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
            Assert.True(module.TryFeedAudio(1, [0.1f]));
            Assert.True(module.TryFeedAudio(1, [0.2f, 0.3f]));
            provider.CompleteStart("session-1");
            await prepare;
            RecognitionOutcome outcome = await module.FinishAsync(
                1,
                "session-1",
                TimeSpan.FromMilliseconds(20),
                CancellationToken.None);

            Assert.Equal(RecognitionOutcomeKind.Completed, outcome.Kind);
            string[] actual = provider.Trace.ToArray();
            expected ??= actual;
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public async Task Finish_WaitsForInFlightAudioDrainBeforeProviderCompletion()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        using var pumpCancellation = new CancellationTokenSource();
        Task pump = module.StartAudioPump(pumpCancellation.Token);
        provider.CompleteStart("session-3");
        await module.PrepareAsync(3, Interlocked.Increment(ref nextOperationId), CancellationToken.None);

        provider.FeedEntered = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        provider.FeedRelease = new ManualResetEventSlim(initialState: false);
        Task feed = Task.Run(() => Assert.True(module.TryFeedAudio(3, [0.4f])));
        await provider.FeedEntered.Task.WaitAsync(TimeSpan.FromSeconds(1));

        Task<RecognitionOutcome> finish = module.FinishAsync(
            3,
            "session-3",
            TimeSpan.FromMilliseconds(10),
            CancellationToken.None);
        await Task.Yield();
        Assert.Equal(0, provider.FinishCount);

        provider.FeedRelease.Set();
        Assert.Equal(RecognitionOutcomeKind.Completed, (await finish).Kind);
        await feed;
        pumpCancellation.Cancel();
        await pump;
        Assert.Equal(["start", "feed:0.4", "finish"], provider.Trace);
    }

    [Fact]
    public async Task Finish_ClosesAudioAdmissionBeforeCallingProviderCompletion()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("session-terminal-admission");
        await module.PrepareAsync(13, Interlocked.Increment(ref nextOperationId), CancellationToken.None);

        var finishEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var finishRelease = new ManualResetEventSlim(initialState: false);
        provider.OnFinish = () =>
        {
            finishEntered.TrySetResult();
            finishRelease.Wait();
        };

        Task<RecognitionOutcome> finish = Task.Run(async () => await module.FinishAsync(
            13,
            "session-terminal-admission",
            TimeSpan.Zero,
            CancellationToken.None));
        await finishEntered.Task.WaitAsync(TimeSpan.FromSeconds(1));

        // A callback that already captured the old turn may arrive after the terminal claim. It is
        // consumed as stale (not misclassified as capacity overflow) and never reaches the provider.
        Assert.True(((IRecognitionAudioPort)module).TryFeedAudio(13, [0.7f]));
        Assert.Equal(["start"], provider.Trace);

        finishRelease.Set();
        Assert.Equal(RecognitionOutcomeKind.Completed, (await finish).Kind);
        Assert.Equal(["start", "finish"], provider.Trace);
    }

    [Fact]
    public async Task Finish_RejectsWrongTurnAndSessionBeforeCallingProvider()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("session-5");
        await module.PrepareAsync(5, Interlocked.Increment(ref nextOperationId), CancellationToken.None);

        await Assert.ThrowsAsync<InvalidOperationException>(() => module.FinishAsync(
            6,
            "session-5",
            TimeSpan.Zero,
            CancellationToken.None));
        await Assert.ThrowsAsync<InvalidOperationException>(() => module.FinishAsync(
            5,
            "wrong-session",
            TimeSpan.Zero,
            CancellationToken.None));

        Assert.Equal(0, provider.FinishCount);
        Assert.Equal(
            RecognitionOutcomeKind.Completed,
            (await module.FinishAsync(
                5,
                "session-5",
                TimeSpan.Zero,
                CancellationToken.None)).Kind);
        Assert.Equal(1, provider.FinishCount);
    }

    [Fact]
    public async Task Finish_InvokesProviderOutsideRecognitionOwnershipLock()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("session-8");
        await module.PrepareAsync(8, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        provider.OnFinish = () =>
        {
            Task lockProbe = Task.Run(() => module.TryGetProviderSession(8, out _, out _));
            Assert.True(
                lockProbe.Wait(TimeSpan.FromSeconds(1)),
                "Recognition provider was invoked while the module ownership lock was held.");
        };

        RecognitionOutcome outcome = await module.FinishAsync(
            8,
            "session-8",
            TimeSpan.Zero,
            CancellationToken.None);

        Assert.Equal(RecognitionOutcomeKind.Completed, outcome.Kind);
    }

    [Fact]
    public void PendingRecognitionAudio_IsBoundedBeforeProviderAdmission()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        for (int i = 0; i < 256; i++)
            Assert.True(module.TryFeedAudio(7, [0.1f]));
        Assert.False(module.TryFeedAudio(7, [0.2f]));

        module.PrepareCaptureTurn(8);
        for (int i = 0; i < 256; i++)
        {
            Assert.Equal(
                SpeakingDecisionKind.None,
                module.ObserveCaptureAudio(
                    8,
                    AssistantState.Listening,
                    [0.1f],
                    default(BargeInValidationCaptureSnapshot)).Kind);
        }

        Assert.Equal(
            SpeakingDecisionKind.RecognitionQueueOverflow,
            module.ObserveCaptureAudio(
                8,
                AssistantState.Listening,
                [0.2f],
                default(BargeInValidationCaptureSnapshot)).Kind);
    }

    [Fact]
    public async Task SessionAudioCapacity_IsSharedByBootstrapAndRegularQueues()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        RecognitionModule? module = null;
        control.OnRecognitionStarted = (turnId, _, sessionId) =>
        {
            for (int index = 0; index < 128; index++)
                Assert.True(module!.TryFeedAudio(turnId, [0.2f]));

            float[][] oversizedBootstrap = Enumerable.Range(0, 129)
                .Select(static _ => new[] { 0.1f })
                .ToArray();
            Assert.False(module!.TryAttachCaptureSession(
                turnId,
                sessionId,
                isValidationTurn: false,
                oversizedBootstrap,
                out bool rejectedSpeechEnded,
                out TimeSpan rejectedSpeechDuration));
            Assert.False(rejectedSpeechEnded);
            Assert.Equal(TimeSpan.Zero, rejectedSpeechDuration);

            float[][] bootstrap = Enumerable.Range(0, 128)
                .Select(static _ => new[] { 0.1f })
                .ToArray();
            Assert.True(module.TryAttachCaptureSession(
                turnId,
                sessionId,
                isValidationTurn: false,
                bootstrap,
                out bool attachedSpeechEnded,
                out TimeSpan attachedSpeechDuration));
            Assert.False(attachedSpeechEnded);
            Assert.Equal(TimeSpan.Zero, attachedSpeechDuration);
            return ValueTask.FromResult(true);
        };
        module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        provider.CompleteStart("session-capacity");
        await module.PrepareAsync(7, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
        RecognitionOutcome outcome = await module.FinishAsync(
            7,
            "session-capacity",
            TimeSpan.Zero,
            CancellationToken.None);

        Assert.Equal(RecognitionOutcomeKind.Completed, outcome.Kind);
        Assert.Equal(256, provider.Trace.Count(entry => entry.StartsWith("feed:", StringComparison.Ordinal)));
    }

    [Fact]
    public async Task CaptureAttach_RejectsOversizedBootstrapBatchBeforeCopyingIt()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("session-oversized-bootstrap");
        await module.PrepareAsync(15, Interlocked.Increment(ref nextOperationId), CancellationToken.None);

        float[][] oversized = Enumerable.Range(0, 257)
            .Select(static _ => new[] { 0.1f })
            .ToArray();
        Assert.False(module.TryAttachCaptureSession(
            15,
            "session-oversized-bootstrap",
            isValidationTurn: false,
            oversized,
            out _,
            out _));
        Assert.Equal(["start"], provider.Trace);
        await module.CleanupAsync(15, "session-oversized-bootstrap");
    }

    [Fact]
    public async Task ConcurrentDuplicateFinishers_ShareOneRegisteredTerminalOperation()
    {
        var provider = new RecordingRecognitionProvider();
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);
        provider.CompleteStart("session-concurrent-finish");
        await module.PrepareAsync(12, Interlocked.Increment(ref nextOperationId), CancellationToken.None);

        var finishEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var finishRelease = new ManualResetEventSlim(initialState: false);
        provider.OnFinish = () =>
        {
            finishEntered.TrySetResult();
            finishRelease.Wait();
        };

        Task<RecognitionOutcome> first = Task.Run(async () => await module.FinishAsync(
            12,
            "session-concurrent-finish",
            TimeSpan.Zero,
            CancellationToken.None));
        await finishEntered.Task.WaitAsync(TimeSpan.FromSeconds(1));

        Task<RecognitionOutcome>[] duplicates = Enumerable.Range(0, 16)
            .Select(_ => module.FinishAsync(
                12,
                "session-concurrent-finish",
                TimeSpan.Zero,
                CancellationToken.None))
            .ToArray();
        Assert.All(duplicates, duplicate => Assert.Same(duplicates[0], duplicate));
        Assert.Equal(1, provider.FinishCount);

        finishRelease.Set();
        RecognitionOutcome[] outcomes = await Task.WhenAll(duplicates.Prepend(first));
        Assert.All(outcomes, outcome => Assert.Equal(RecognitionOutcomeKind.Completed, outcome.Kind));
        Assert.Equal(1, provider.FinishCount);
    }

    [Fact]
    public async Task CompletedRecognitionSessions_AreRetiredAndTerminalDeduplicationIsBounded()
    {
        var provider = new RecordingRecognitionProvider();
        provider.ImmediateSessionId = "reused-provider-session";
        var control = new RecordingRecognitionControlPort();
        long nextOperationId = 0;
        var module = new RecognitionModule(
            provider,
            control,
            NullLogger.Instance);

        for (long turnId = 1; turnId <= 300; turnId++)
        {
            await module.PrepareAsync(turnId, Interlocked.Increment(ref nextOperationId), CancellationToken.None);
            RecognitionOutcome outcome = await module.FinishAsync(
                turnId,
                "reused-provider-session",
                TimeSpan.Zero,
                CancellationToken.None);
            Assert.Equal(RecognitionOutcomeKind.Completed, outcome.Kind);
        }

        Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByTurn"));
        Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByProviderId"));
        Assert.Equal(0, PrivateCollectionCount(module, "_pendingAudioByTurn"));
        Assert.InRange(PrivateCollectionCount(module, "_terminalOperations"), 1, 256);
        Assert.Equal(256, PrivateCollectionCount(module, "_completedTerminalOperations"));
    }

    [Fact]
    public async Task AudioPump_ProviderFeedFailureTerminatesOnlyItsSessionAndKeepsConsuming()
    {
        var provider = new RecordingRecognitionProvider { ImmediateSessionId = "failed-session" };
        var failed = new TaskCompletionSource<RecognitionOutcome>(TaskCreationOptions.RunContinuationsAsynchronously);
        var control = new RecordingRecognitionControlPort { OnRecognitionFailed = outcome => failed.TrySetResult(outcome) };
        var module = new RecognitionModule(provider, control, NullLogger.Instance);
        using var cancellation = new CancellationTokenSource();
        Task pump = module.StartAudioPump(cancellation.Token);
        await module.PrepareAsync(1, 1, CancellationToken.None);
        provider.OnFeed = audio =>
        {
            if (audio[0] == 1f)
            {
                Assert.True(module.TryFeedAudio(1, [2f]));
                Assert.True(module.TryFeedAudio(1, [3f]));
            }
            if (audio[0] == 2f)
                throw new InvalidOperationException("injected queued feed failure");
        };
        try
        {
            Assert.True(module.TryFeedAudio(1, [1f]));
            RecognitionOutcome failure = await failed.Task.WaitAsync(TimeSpan.FromSeconds(1));
            Assert.Equal(RecognitionOutcomeKind.Failed, failure.Kind);
            Assert.Equal(1, failure.TurnId);
            Assert.Equal("audio-feed-failed", failure.Failure.MessageKey);
            Assert.False(pump.IsCompleted);
            Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByTurn"));
            Assert.Equal(1, provider.FinishCount);

            provider.ImmediateSessionId = "healthy-session";
            await module.PrepareAsync(2, 2, CancellationToken.None);
            var queuedFeed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            provider.OnFeed = audio =>
            {
                if (audio[0] == 4f)
                    Assert.True(module.TryFeedAudio(2, [5f]));
                if (audio[0] == 5f)
                    queuedFeed.TrySetResult();
            };
            Assert.True(module.TryFeedAudio(2, [4f]));
            await queuedFeed.Task.WaitAsync(TimeSpan.FromSeconds(1));
            Assert.Equal(RecognitionOutcomeKind.Completed,
                (await module.FinishAsync(2, "healthy-session", TimeSpan.Zero, CancellationToken.None)).Kind);
        }
        finally
        {
            cancellation.Cancel();
            try { await pump; } catch (InvalidOperationException) { }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Finish_AfterFeedFailureReleasesProviderAndNeverCommitsPartialRecognition(bool directFailure)
    {
        var provider = new RecordingRecognitionProvider { ImmediateSessionId = "feed-failure" };
        var module = new RecognitionModule(provider, new RecordingRecognitionControlPort(), NullLogger.Instance);
        await module.PrepareAsync(1, 1, CancellationToken.None);
        provider.OnFeed = audio =>
        {
            if (directFailure || audio[0] == 2f)
                throw new InvalidOperationException("injected feed failure");
            Assert.True(module.TryFeedAudio(1, [2f]));
        };
        if (directFailure)
            Assert.Throws<InvalidOperationException>(() => module.TryFeedAudio(1, [1f]));
        else
            Assert.True(module.TryFeedAudio(1, [1f]));

        RecognitionOutcome outcome = await module.FinishAsync(1, "feed-failure", TimeSpan.Zero, CancellationToken.None);
        Assert.Equal(RecognitionOutcomeKind.Failed, outcome.Kind);
        Assert.Null(outcome.RawText);
        Assert.Equal(1, provider.FinishCount);
        Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByTurn"));
        Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByProviderId"));
        Assert.Equal(outcome, await module.FinishAsync(1, "feed-failure", TimeSpan.Zero, CancellationToken.None));
        Assert.Equal(1, provider.FinishCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Finish_CancellationBeforeProviderFinishStillReleasesSession(bool feedInFlight)
    {
        var provider = new RecordingRecognitionProvider { ImmediateSessionId = "cancelled-session" };
        var module = new RecognitionModule(provider, new RecordingRecognitionControlPort(), NullLogger.Instance);
        await module.PrepareAsync(1, 1, CancellationToken.None);
        using var cancellation = new CancellationTokenSource();
        using var feedRelease = new ManualResetEventSlim(false);
        Task? feed = null;
        if (feedInFlight)
        {
            provider.FeedEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            provider.FeedRelease = feedRelease;
            feed = Task.Run(() => Assert.True(module.TryFeedAudio(1, [1f])));
            await provider.FeedEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        }
        cancellation.Cancel();

        try
        {
            Task<RecognitionOutcome> finish = module.FinishAsync(1, "cancelled-session", TimeSpan.Zero, cancellation.Token);
            if (feedInFlight)
            {
                Assert.Equal(0, provider.FinishCount);
                feedRelease.Set();
                await feed!;
            }
            RecognitionOutcome outcome = await finish.WaitAsync(TimeSpan.FromSeconds(2));
            Assert.Equal(RecognitionOutcomeKind.Cancelled, outcome.Kind);
            Assert.Null(outcome.RawText);
            Assert.Equal(1, provider.FinishCount);
            Assert.Equal(0, PrivateCollectionCount(module, "_sessionsByProviderId"));
            Assert.Equal(outcome, await module.FinishAsync(1, "cancelled-session", TimeSpan.Zero, CancellationToken.None));
            await module.CleanupAsync(1, "cancelled-session");
            Assert.Equal(1, provider.FinishCount);
        }
        finally
        {
            feedRelease.Set();
            if (feed is not null)
                await feed;
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Finish_CancelledProviderWaitReleasesRetainedSessionWithoutRetryingFinish(bool recordAudio)
    {
        using var provider = new RetainingCancellationAsrService();
        string directory = Path.Combine(Path.GetTempPath(), "voicebox-cancelled-asr-" + Guid.NewGuid().ToString("N"));
        using var recorder = recordAudio
            ? new RecognitionAudioRecordingService(provider, directory, 16000, 1, NullLogger.Instance)
            : null;
        var module = new RecognitionModule(
            new OfflineRecognitionProviderAdapter(recorder is null ? provider : recorder),
            new RecordingRecognitionControlPort(), NullLogger.Instance);
        try
        {
            for (int turn = 1; turn <= 3; turn++)
            {
                using var cancellation = new CancellationTokenSource();
                await module.PrepareAsync(turn, turn, CancellationToken.None);
                Assert.True(module.TryFeedAudio(turn, [0.1f]));
                string sessionId = provider.LastSessionId;
                Task<RecognitionOutcome> finish = module.FinishAsync(turn, sessionId, TimeSpan.Zero, cancellation.Token);
                await provider.FinishEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
                cancellation.Cancel();

                RecognitionOutcome outcome = await finish.WaitAsync(TimeSpan.FromSeconds(2));
                Assert.Equal(RecognitionOutcomeKind.Cancelled, outcome.Kind);
                Assert.Equal(0, provider.ActiveSessions);
                Assert.Equal(turn, provider.FinishCount);
                Assert.Equal(turn, provider.CancelCount);
                Assert.Equal(outcome, await module.FinishAsync(turn, sessionId, TimeSpan.Zero, CancellationToken.None));
                await module.CleanupAsync(turn, sessionId);
                Assert.Equal(turn, provider.CancelCount);
            }
            if (recordAudio)
            {
                Assert.Equal(3, Directory.GetFiles(directory, "*.wav").Length);
                Assert.Equal(3, Directory.GetFiles(directory, "*.json").Length);
            }
        }
        finally
        {
            if (Directory.Exists(directory))
                Directory.Delete(directory, recursive: true);
        }
    }

    // Confucius4 retains a cancelled Finish waiter for retry and limits active sessions.
    // The fake exercises that documented contract without loading an ONNX model.
    private sealed class RetainingCancellationAsrService : IASRService
    {
        public int ActiveSessions { get; private set; }
        public int FinishCount { get; private set; }
        public int CancelCount { get; private set; }
        public string LastSessionId { get; private set; } = "";
        public TaskCompletionSource FinishEntered { get; private set; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Task<SessionInfo> StartSessionAsync(CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            if (ActiveSessions != 0)
                throw new InvalidOperationException("session capacity exhausted");
            ActiveSessions++;
            LastSessionId = "session-" + (FinishCount + 1);
            FinishEntered = new(TaskCreationOptions.RunContinuationsAsynchronously);
            return Task.FromResult(new SessionInfo(LastSessionId));
        }

        public void FeedAudio(string sessionId, ReadOnlySpan<float> audioData) { }

        public async Task<AsrResult?> FinishSessionAsync(string sessionId, CancellationToken ct = default)
        {
            FinishCount++;
            FinishEntered.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, ct);
            return null;
        }

        public Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            Assert.Equal(LastSessionId, sessionId);
            CancelCount++;
            ActiveSessions--;
            return Task.CompletedTask;
        }

        public void Dispose() { }
    }

    private static int PrivateCollectionCount(object owner, string fieldName)
    {
        object value = owner.GetType()
            .GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(owner)!;
        return (int)value.GetType().GetProperty("Count")!.GetValue(value)!;
    }

    private static Type Require(Assembly assembly, string name)
        => assembly.GetType(name)
            ?? throw new Xunit.Sdk.XunitException($"Required recognition contract type '{name}' was not found.");

    private sealed class RecordingRecognitionProvider : IRecognitionProviderAdapter
    {
        private readonly TaskCompletionSource<string> _start =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public string? ImmediateSessionId { get; set; }

        public int StartCount { get; private set; }

        public int FinishCount { get; private set; }

        public List<string> Trace { get; } = [];

        public TaskCompletionSource? FeedEntered { get; set; }

        public ManualResetEventSlim? FeedRelease { get; set; }

        public Action? OnFinish { get; set; }

        public Action<float[]>? OnFeed { get; set; }

        public async Task<string> StartSessionAsync(CancellationToken cancellationToken)
        {
            StartCount++;
            Trace.Add("start");
            if (ImmediateSessionId is not null)
                return ImmediateSessionId;
            return await _start.Task.WaitAsync(cancellationToken);
        }

        public void FeedAudio(string sessionId, ReadOnlySpan<float> audio)
        {
            FeedEntered?.TrySetResult();
            FeedRelease?.Wait();
            Trace.Add($"feed:{string.Join(',', audio.ToArray())}");
            OnFeed?.Invoke(audio.ToArray());
        }

        public Task<string?> FinishSessionAsync(
            string sessionId,
            CancellationToken cancellationToken)
        {
            FinishCount++;
            OnFinish?.Invoke();
            Trace.Add("finish");
            return Task.FromResult<string?>(" raw text ");
        }

        public void CompleteStart(string sessionId) => _start.TrySetResult(sessionId);

        public void FailStart(Exception exception) => _start.TrySetException(exception);
    }

    private sealed class RecordingRecognitionControlPort : IRecognitionControlSignalPort
    {
        public List<(long TurnId, long OperationId, string SessionId)> Ready { get; } = [];

        public List<RecognitionOutcome> Failed { get; } = [];

        public Action<RecognitionOutcome>? OnRecognitionFailed { get; set; }

        public Func<long, long, string, ValueTask<bool>>? OnRecognitionStarted { get; set; }

        public ValueTask<bool> PublishRecognitionStartedAsync(
            long turnId,
            long operationId,
            string sessionId,
            CancellationToken cancellationToken)
        {
            Ready.Add((turnId, operationId, sessionId));
            return OnRecognitionStarted?.Invoke(turnId, operationId, sessionId)
                ?? ValueTask.FromResult(true);
        }

        public ValueTask PublishRecognitionFailedAsync(
            RecognitionOutcome outcome,
            CancellationToken cancellationToken)
        {
            Failed.Add(outcome);
            OnRecognitionFailed?.Invoke(outcome);
            return ValueTask.CompletedTask;
        }
    }

    private sealed class StaticTurnStateView(TurnSnapshot snapshot) : ITurnStateView
    {
        public TurnSnapshot Snapshot { get; } = snapshot;

        public AssistantState ConversationState => Snapshot.ConversationState;

        public long TurnId => Snapshot.TurnId;

        public long NextTurnOrdinal => Snapshot.NextTurnOrdinal;

        public long ValidationTurnId => Snapshot.ValidationTurnId;

        public bool IsCurrentTurn(long turnId) => Snapshot.TurnId == turnId;
    }

    private sealed class StaticValidationTurnView(long turnId) : IBargeInValidationTurnView
    {
        public BargeInValidationCaptureSnapshot CaptureSnapshot => new(turnId, 0, null);

        public long ValidationTurnId => turnId;

        public long InterruptedTurnId => 0;

        public CancellationTokenSource? ResponseCts => null;

        public bool IsValidating(long candidateTurnId) => candidateTurnId == turnId;
    }

    private sealed class EmptyValidationPreRoll : IValidationCapturePreRollPort
    {
        public bool TryTakePendingAudio(out List<float[]> audio)
        {
            audio = [];
            return false;
        }
    }

    private sealed class EmptyCandidatePreRoll : ICandidatePreRollSource
    {
        public void DrainPendingAudioInto(List<float[]> destination)
        {
        }
    }
}
