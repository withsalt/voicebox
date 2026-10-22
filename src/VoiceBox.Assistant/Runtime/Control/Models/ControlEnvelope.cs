namespace VoiceBox.Assistant.Runtime.Control.Models;

internal readonly record struct ControlEnvelope
{
    private ControlEnvelope(
        long ingressSequence,
        long sourceTimestamp,
        AssistantSignalSource source,
        ControlEnvelopeKind kind,
        RuntimeCommand command,
        AssistantSignal signal,
        AssistantFailure failure)
    {
        IngressSequence = ingressSequence;
        SourceTimestamp = sourceTimestamp;
        Source = source;
        Kind = kind;
        Command = command;
        Signal = signal;
        Failure = failure;
    }

    public long IngressSequence { get; }

    public long SourceTimestamp { get; }

    public AssistantSignalSource Source { get; }

    public ControlEnvelopeKind Kind { get; }

    public RuntimeCommand Command { get; }

    public AssistantSignal Signal { get; }

    public AssistantFailure Failure { get; }

    public static ControlEnvelope FromCommand(
        long ingressSequence,
        long sourceTimestamp,
        AssistantSignalSource source,
        RuntimeCommand command)
        => new(
            ingressSequence,
            sourceTimestamp,
            source,
            ControlEnvelopeKind.RuntimeCommand,
            command,
            default,
            default);

    public static ControlEnvelope FromSignal(
        long ingressSequence,
        long sourceTimestamp,
        AssistantSignalSource source,
        AssistantSignal signal)
        => new(
            ingressSequence,
            sourceTimestamp,
            source,
            ControlEnvelopeKind.AssistantSignal,
            default,
            signal.WithIngressMetadata(ingressSequence, sourceTimestamp),
            default);

    public static ControlEnvelope FromTerminalFault(
        long ingressSequence,
        long sourceTimestamp,
        AssistantFailure failure)
        => new(
            ingressSequence,
            sourceTimestamp,
            AssistantSignalSource.TerminalFault,
            ControlEnvelopeKind.TerminalFault,
            default,
            default,
            failure);
}
