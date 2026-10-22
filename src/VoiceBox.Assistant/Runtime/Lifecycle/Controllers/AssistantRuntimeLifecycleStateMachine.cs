using VoiceBox.Assistant.Runtime.Lifecycle.Models;

namespace VoiceBox.Assistant.Runtime.Lifecycle.Controllers;

/// <summary>
/// State-only lifecycle boundary. The Runtime remains responsible for public state publication;
/// this type never enters the Turn state machine.
/// </summary>
internal sealed class AssistantRuntimeLifecycleStateMachine
{
    public AssistantRuntimeLifecycleState State { get; private set; } = AssistantRuntimeLifecycleState.Created;

    public bool BeginStart()
    {
        // VoiceAssistant's public contract is single-start. A stopped instance cannot be
        // started again: run-scoped workers such as the control loop and recognition pump are
        // intentionally one-shot and must never be partially restarted before failing.
        if (State != AssistantRuntimeLifecycleState.Created)
            return false;

        State = AssistantRuntimeLifecycleState.Starting;
        return true;
    }

    public bool CompleteStart()
    {
        if (State != AssistantRuntimeLifecycleState.Starting)
            return false;

        State = AssistantRuntimeLifecycleState.Running;
        return true;
    }

    public bool BeginStop()
    {
        if (State is not (AssistantRuntimeLifecycleState.Starting or
            AssistantRuntimeLifecycleState.Running or
            AssistantRuntimeLifecycleState.Faulted))
            return false;

        State = AssistantRuntimeLifecycleState.Stopping;
        return true;
    }

    public void CompleteStop()
    {
        switch (State)
        {
            case AssistantRuntimeLifecycleState.Starting:
            case AssistantRuntimeLifecycleState.Running:
            case AssistantRuntimeLifecycleState.Stopping:
                State = AssistantRuntimeLifecycleState.Stopped;
                break;
        }
    }

    public void FailStart()
    {
        if (State == AssistantRuntimeLifecycleState.Starting)
            State = AssistantRuntimeLifecycleState.Stopped;
    }

    public bool MarkFaulted()
    {
        if (State is AssistantRuntimeLifecycleState.Disposed or AssistantRuntimeLifecycleState.Faulted)
            return false;

        State = AssistantRuntimeLifecycleState.Faulted;
        return true;
    }

    public void MarkDisposed() => State = AssistantRuntimeLifecycleState.Disposed;
}
