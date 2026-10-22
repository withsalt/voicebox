using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Wakeup.Contracts;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Runtime.Lifecycle.Models;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// Executes the immutable lifecycle topology through closed typed participants. It owns no
/// composition graph and has no capability lookup or dynamic registration surface.
/// </summary>
internal sealed class AssistantRuntimeLifecycleCoordinator
{
    private readonly AssistantRuntimeLifecycleStateMachine _lifecycle = new();
    private readonly ModuleLifecyclePlan _plan;
    private readonly IReadOnlyList<ModuleOwnership> _ownership;
    private readonly IAssistantAudioFrontend _audio;
    private readonly IWakeupModule? _wakeup;
    private readonly ISpeakerModule? _speaker;
    private readonly IPlaybackBargeInDetector? _bargeIn;
    private readonly AssistantTaskSupervisor _tasks;
    private readonly ILogger _logger;
    private readonly IResponseGenerator? _responseGeneration;
    private readonly List<AssistantLifecycleParticipant> _started = [];
    private int _assistantOwnedDisposed;

    public AssistantRuntimeLifecycleCoordinator(
        ModuleLifecyclePlan plan,
        IReadOnlyList<ModuleOwnership> ownership,
        IAssistantAudioFrontend audio,
        IWakeupModule? wakeup,
        ISpeakerModule? speaker,
        IPlaybackBargeInDetector? bargeIn,
        AssistantTaskSupervisor tasks,
        ILogger logger,
        IResponseGenerator? responseGeneration = null)
    {
        _plan = plan ?? throw new ArgumentNullException(nameof(plan));
        _ownership = ownership ?? throw new ArgumentNullException(nameof(ownership));
        _audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _wakeup = wakeup;
        _speaker = speaker;
        _bargeIn = bargeIn;
        _tasks = tasks ?? throw new ArgumentNullException(nameof(tasks));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _responseGeneration = responseGeneration;
    }

    /// <summary>
    /// Runtime lifecycle state has a single writer: this coordinator. The shell asks for
    /// transitions; it never mutates the state machine directly.
    /// </summary>
    public AssistantRuntimeLifecycleState State => _lifecycle.State;

    /// <summary>Exposed so fault convergence can mark the same state machine under the shared lock.</summary>
    public AssistantRuntimeLifecycleStateMachine StateMachine => _lifecycle;

    public bool BeginStart() => _lifecycle.BeginStart();

    public bool CompleteStart() => _lifecycle.CompleteStart();

    public bool BeginStop() => _lifecycle.BeginStop();

    public void CompleteStop() => _lifecycle.CompleteStop();

    public void FailStart() => _lifecycle.FailStart();

    public void MarkDisposed() => _lifecycle.MarkDisposed();

    public async Task RollbackStartupAsync()
    {
        await RollbackStartedAsync().ConfigureAwait(false);
        await _tasks.WaitForCompletionAsync().ConfigureAwait(false);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _started.Clear();
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        try
        {
            foreach (AssistantLifecycleParticipant participant in _plan.StartOrder)
            {
                cancellationToken.ThrowIfCancellationRequested();
                object identity = GetIdentity(participant);
                if (!seen.Add(identity))
                    continue;

                await StartParticipantAsync(participant, cancellationToken).ConfigureAwait(false);
                _started.Add(participant);
            }
        }
        catch
        {
            await RollbackStartedAsync().ConfigureAwait(false);
            throw;
        }
    }

    public async Task StopAsync()
    {
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        foreach (AssistantLifecycleParticipant participant in _plan.StartOrder.Reverse())
        {
            object identity = GetIdentity(participant);
            if (!seen.Add(identity))
                continue;

            await StopParticipantAsync(participant, logFailures: true).ConfigureAwait(false);
        }

        await _tasks.WaitForCompletionAsync().ConfigureAwait(false);
        _started.Clear();
    }

    /// <summary>
    /// Releases only the participants the build-time ownership plan marks as AssistantOwned.
    /// Borrowed providers are never disposed here, and each identity is disposed at most once.
    /// </summary>
    public void DisposeAssistantOwned()
    {
        if (Interlocked.Exchange(ref _assistantOwnedDisposed, 1) != 0)
            return;

        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        foreach (AssistantLifecycleParticipant participant in _plan.StartOrder.Reverse())
        {
            object identity = GetIdentity(participant);
            if (!seen.Add(identity))
                continue;
            if (!IsAssistantOwned(participant))
                continue;

            switch (participant)
            {
                case AssistantLifecycleParticipant.AudioFrontend:
                    // Only the Assistant-created split adapter owns disposable state; the
                    // borrowed record/play providers are never disposed here.
                    TryDispose(_audio as IDisposable);
                    break;
                case AssistantLifecycleParticipant.WakeupGate:
                    TryDispose(_wakeup);
                    break;
                case AssistantLifecycleParticipant.SpeakerEvidence:
                    TryDispose(_speaker);
                    break;
                case AssistantLifecycleParticipant.BargeIn:
                    TryDispose(_bargeIn);
                    break;
                case AssistantLifecycleParticipant.ResponseGeneration:
                    TryDispose(_responseGeneration as IDisposable);
                    break;
            }
        }
    }

    private bool IsAssistantOwned(AssistantLifecycleParticipant participant)
    {
        string moduleId = participant switch
        {
            AssistantLifecycleParticipant.AudioFrontend => "AudioFrontend.Adapter",
            AssistantLifecycleParticipant.WakeupGate => "Wakeup.Module",
            AssistantLifecycleParticipant.SpeakerEvidence => "Speaker.Module",
            AssistantLifecycleParticipant.BargeIn => "BargeIn.Module",
            AssistantLifecycleParticipant.ResponseGeneration => "StructuredToolUse.Runtime",
            _ => throw new ArgumentOutOfRangeException(nameof(participant), participant, null),
        };

        foreach (ModuleOwnership ownership in _ownership)
        {
            if (string.Equals(ownership.ModuleId, moduleId, StringComparison.Ordinal))
                return ownership.Ownership == ResourceOwnership.AssistantOwned;
        }

        return false;
    }

    private async Task RollbackStartedAsync()
    {
        foreach (AssistantLifecycleParticipant participant in _started.AsEnumerable().Reverse())
            // One provider's stop failure must not skip the remaining started participants
            // or replace the exception that caused startup to fail.
            await StopParticipantAsync(participant, logFailures: true).ConfigureAwait(false);

        _started.Clear();
    }

    private object GetIdentity(AssistantLifecycleParticipant participant)
        => participant switch
        {
            AssistantLifecycleParticipant.AudioFrontend => _audio,
            AssistantLifecycleParticipant.WakeupGate => _wakeup
                ?? throw new InvalidOperationException("Lifecycle plan includes WakeupGate without a wakeup module."),
            AssistantLifecycleParticipant.SpeakerEvidence => _speaker
                ?? throw new InvalidOperationException("Lifecycle plan includes SpeakerEvidence without a speaker module."),
            AssistantLifecycleParticipant.BargeIn => _bargeIn
                ?? throw new InvalidOperationException("Lifecycle plan includes BargeIn without a barge-in module."),
            AssistantLifecycleParticipant.ResponseGeneration => _responseGeneration
                ?? throw new InvalidOperationException("Lifecycle plan includes ResponseGeneration without a response generator."),
            _ => throw new ArgumentOutOfRangeException(nameof(participant), participant, null),
        };

    private async Task StartParticipantAsync(
        AssistantLifecycleParticipant participant,
        CancellationToken cancellationToken)
    {
        switch (participant)
        {
            case AssistantLifecycleParticipant.AudioFrontend:
                await _audio.StartAsync(cancellationToken).ConfigureAwait(false);
                break;
            case AssistantLifecycleParticipant.WakeupGate:
                _wakeup!.Start();
                break;
            case AssistantLifecycleParticipant.SpeakerEvidence:
                _speaker!.Start();
                break;
            case AssistantLifecycleParticipant.BargeIn:
                _bargeIn!.ResetStream();
                break;
            case AssistantLifecycleParticipant.ResponseGeneration:
                if (_responseGeneration is not IResponseGeneratorLifecycle lifecycle)
                {
                    throw new InvalidOperationException(
                        "Structured response generator does not implement its lifecycle contract.");
                }
                await lifecycle.StartAsync(cancellationToken).ConfigureAwait(false);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(participant), participant, null);
        }
    }

    private async Task StopParticipantAsync(AssistantLifecycleParticipant participant, bool logFailures)
    {
        try
        {
            switch (participant)
            {
                case AssistantLifecycleParticipant.AudioFrontend:
                    await _audio.StopAsync().ConfigureAwait(false);
                    break;
                case AssistantLifecycleParticipant.WakeupGate:
                    _wakeup!.Stop();
                    break;
                case AssistantLifecycleParticipant.SpeakerEvidence:
                    await _speaker!.StopAsync().ConfigureAwait(false);
                    break;
                case AssistantLifecycleParticipant.BargeIn:
                    _bargeIn!.ResetStream();
                    break;
                case AssistantLifecycleParticipant.ResponseGeneration:
                    if (_responseGeneration is IResponseGeneratorLifecycle lifecycle)
                        await lifecycle.StopAsync().ConfigureAwait(false);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(participant), participant, null);
            }
        }
        catch (ObjectDisposedException) when (participant == AssistantLifecycleParticipant.BargeIn)
        {
        }
        catch (Exception ex) when (logFailures)
        {
            _logger.LogWarning(ex, "Stop lifecycle participant {Participant} failed.", participant);
        }
    }

    private static void TryDispose(IDisposable? value)
    {
        try { value?.Dispose(); } catch { }
    }
}
