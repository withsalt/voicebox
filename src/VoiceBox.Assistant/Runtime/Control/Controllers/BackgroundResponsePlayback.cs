using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Kernel.Models;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Response.Controllers;
using VoiceBox.Assistant.Runtime.Control.Models;
using VoiceBox.Assistant.Runtime.Control.State;
using VoiceBox.Assistant.Runtime.Effects.Contracts;

namespace VoiceBox.Assistant.Runtime.Control.Controllers;

/// <summary>Admits a completed background result as a fresh, interruptible response turn.</summary>
internal sealed class BackgroundResponsePlayback(
    TurnSnapshotStore snapshots,
    AssistantControlSignalGateway control,
    ResponseModule responses,
    IResponseTurnResourcePort resources) : IBackgroundResponsePlaybackPort
{
    public async Task PlayAsync(string text, CancellationToken cancellationToken)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(
            cancellationToken, resources.RunToken);
        CancellationToken token = cancellation.Token;
        while (true)
        {
            token.ThrowIfCancellationRequested();
            TurnSnapshot observed = snapshots.Snapshot;
            if (observed.ConversationState != AssistantState.Idle || observed.ValidationTurnId != 0)
            {
                await snapshots.WaitForChangeAsync(observed, token).ConfigureAwait(false);
                continue;
            }

            long turnId = observed.NextTurnOrdinal;
            long operationId = control.NextOperationId();
            // Once admitted, observe the verdict even if the caller cancels, so an accepted
            // turn always runs its terminal cleanup instead of being stranded in Thinking.
            AssistantControlSignalResult admission = await control.EnqueueAsync(
                AssistantSignal.BackgroundResponseStarted(turnId, operationId),
                AssistantSignalSource.Response,
                CancellationToken.None).ConfigureAwait(false);
            if (!admission.Accepted)
                continue;

            try
            {
                await responses.RunBackgroundTextTurnAsync(
                    turnId, operationId, text, token, resources).ConfigureAwait(false);
            }
            catch (ResponseModule.ResponseOperationAdmissionRejectedException)
            {
                // A user interruption may have won before the response attached its resources.
            }
            finally
            {
                try
                {
                    await resources.PublishTurnCompletedAsync(turnId, false).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (resources.IsShuttingDown) { }
                catch (InvalidOperationException) when (resources.IsShuttingDown) { }
                finally
                {
                    responses.ReleaseOperationContext(turnId);
                }
            }
            return;
        }
    }
}
