using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.TurnBoundary.Contracts;
using VoiceBox.Assistant.Modules.TurnBoundary.Models;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Control.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Theory]
    [InlineData(320)]
    [InlineData(640)]
    public async Task ActiveNearEndCapture_RebasedConfirmationDoesNotInheritVadSpeech(long selectedEndSample)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var endedTurns = new List<long>();
        var coordinator = CreateDataPlaneCoordinator(services, faults, (turnId, _) => endedTurns.Add(turnId));
        ObserveRecognitionAudio(coordinator, 0, new float[selectedEndSample]);
        var lease = CreateConfirmationLease(selectedEndSample) with { EndpointVadCandidate = true };
        coordinator.SeedPendingConfirmation(lease);
        Assert.True(coordinator.TryAcquire(1, out NearEndConfirmationLease taken));

        // The old VAD utterance has no endpoint after the stream restarts. Even when its old
        // boundary happens to equal the new end sample, its speech state belongs to epoch 1.
        ObserveRecognitionAudio(coordinator, 0, new float[320], streamEpoch: 2);
        Assert.True(coordinator.Confirmations.TryCommit(in taken, CreateConfirmedDecision()));
        await coordinator.AttachAsrSessionAsync(1, "rebased-vad-capture");

        for (int frame = 1; frame <= 20; frame++)
        {
            var duplex = CreateDuplexFrame(frame + 1, frame * 320, 320) with { StreamEpoch = 2 };
            coordinator.ObserveDuplexAudio(duplex, playbackStartupGuardActive: false);
            FeedRecognitionAudio(coordinator, frame * 320, new float[320], streamEpoch: 2);
        }

        Assert.Empty(faults);
        Assert.Equal([1L], endedTurns);
        Assert.Equal(0, coordinator.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.Equal(320, services.Asr.FedAudio["rebased-vad-capture"].Sum(static chunk => chunk.Length));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ActiveNearEndConfirmation_PlaybackBoundaryInvalidatesPendingAndAcquiredLeases(bool acquired)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        ObserveRecognitionAudio(coordinator, 0, new float[320]);
        coordinator.SeedPendingConfirmation(CreateConfirmationLease(320));
        NearEndConfirmationLease taken = default;
        if (acquired)
            Assert.True(coordinator.TryAcquire(1, out taken));

        coordinator.ClearAudioReservoirs();

        Assert.False(coordinator.GetDiagnosticSnapshot().ConfirmationPending);
        Assert.False(coordinator.TryAcquire(1, out _));
        Assert.Equal(0, coordinator.GetDiagnosticSnapshot().ActiveTurnId);

        ObserveRecognitionAudio(coordinator, 320, new float[320]);
        var next = CreateConfirmationLease(640) with { OperationId = 2 };
        coordinator.SeedPendingConfirmation(next);
        if (acquired)
        {
            Assert.False(coordinator.Confirmations.TryCommit(in taken, CreateConfirmedDecision()));
            coordinator.Confirmations.Abort(in taken);
        }
        Assert.True(coordinator.TryAcquire(2, out NearEndConfirmationLease current));
        Assert.True(coordinator.Confirmations.TryCommit(in current, CreateConfirmedDecision()));
        Assert.Equal(1, coordinator.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.Empty(faults);
    }

    [Theory]
    [InlineData(2, 2)]
    [InlineData(1, 2)]
    public void ActiveNearEndConfirmation_CannotInterruptADifferentOwner(long leaseOwner, long interruptedOwner)
    {
        var services = TestServices.Create();
        var faults = new List<AssistantFailure>();
        var coordinator = CreateDataPlaneCoordinator(services, faults);
        ObserveRecognitionAudio(coordinator, 0, new float[320]);
        var lease = CreateConfirmationLease(320) with { OwnerTurnId = leaseOwner };
        coordinator.SeedPendingConfirmation(lease);
        Assert.True(coordinator.TryAcquire(1, out NearEndConfirmationLease taken));
        var confirmed = CreateConfirmedDecision() with { TurnId = 3, InterruptedTurnId = interruptedOwner };

        Assert.False(coordinator.Confirmations.TryCommit(in taken, in confirmed));

        Assert.Equal(0, coordinator.GetDiagnosticSnapshot().ActiveTurnId);
        Assert.Empty(faults);
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task ActiveNearEndConfirmation_ControlRequiresCurrentOwnerAndSuccessfulCommit(
        bool ownerMatches,
        bool commitSucceeds)
    {
        var services = TestServices.Create();
        services.Asr.CompleteStart("session-1");
        services.Tts.BlockAfterFirstChunk = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var assistant = services.CreateAssistant(enableBargeIn: true);
        await assistant.StartAsync();
        try
        {
            services.Vad.EmitSpeechStarted();
            services.Vad.EmitSpeaking([0.1f, 0.2f, 0.3f]);
            services.Vad.EmitSpeechEnded(TimeSpan.FromMilliseconds(120));
            await WaitUntilAsync(() => assistant.State == AssistantState.Speaking && services.Audio.AcceptedSamples > 0);

            AssistantRuntime runtime = GetPrivateField<AssistantRuntime>(assistant, "_runtime");
            var turns = GetPrivateField<BargeInTurnCoordinator>(runtime, "_bargeInTurns");
            var lease = CreateConfirmationLease(320) with
            {
                OwnerTurnId = ownerMatches ? turns.CurrentTurnId : turns.CurrentTurnId + 1,
                InitialAudio = [new float[320]],
                Decision = default(BargeInDecision) with { Action = BargeInAction.Interrupt },
            };
            var confirmations = new TestNearEndConfirmationStore(lease, commitSucceeds);
            using var ingress = new AssistantControlIngress();
            var handler = new ActiveNearEndControlHandler(
                ingress,
                confirmations,
                turns,
                new TestRecognitionAudioPort(services.Asr),
                GetPrivateField<ITurnBoundaryEndpointSink>(runtime, "_speechBoundary"),
                benchmarkProbe: null);

            BargeInSemanticPlan plan = handler.ProcessInterruptionRequested(lease.OperationId);

            Assert.Equal(
                ownerMatches && commitSucceeds ? BargeInSemanticPlanKind.InterruptionRequested : BargeInSemanticPlanKind.None,
                plan.Kind);
            Assert.Equal(ownerMatches, confirmations.CommitAttempted);
            Assert.Equal(!ownerMatches || !commitSucceeds, confirmations.Aborted);
            Assert.Equal(AssistantState.Speaking, assistant.State);
            Assert.Equal(0, services.Audio.ClearQueueCount);
        }
        finally
        {
            services.Tts.BlockAfterFirstChunk.TrySetResult();
            await assistant.StopAsync();
        }
    }

    private sealed class TestNearEndConfirmationStore(
        NearEndConfirmationLease lease,
        bool commitSucceeds) : INearEndConfirmationStore
    {
        public bool CommitAttempted { get; private set; }

        public bool Aborted { get; private set; }

        public ActiveNearEndDiagnosticSnapshot GetDiagnosticSnapshot() => default;

        public bool TryAcquire(long operationId, out NearEndConfirmationLease acquired)
        {
            acquired = lease;
            return operationId == lease.OperationId;
        }

        public bool TryCommit(in NearEndConfirmationLease acquired, in SpeakingDecision confirmed)
        {
            CommitAttempted = true;
            return commitSucceeds;
        }

        public void Abort(in NearEndConfirmationLease acquired) => Aborted = true;
    }
}
