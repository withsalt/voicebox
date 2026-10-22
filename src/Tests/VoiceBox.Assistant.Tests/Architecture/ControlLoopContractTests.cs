using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Composition.Wiring;
using VoiceBox.Assistant.Modules.BargeIn.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.Conversation;
using VoiceBox.Assistant.Modules.Conversation.Contracts;
using VoiceBox.Assistant.Modules.Recognition.Controllers;
using VoiceBox.Assistant.Modules.Response.Controllers;
using VoiceBox.Assistant.Modules.TurnBoundary.Controllers;
using VoiceBox.Assistant.Modules.Wakeup.Controllers;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Handlers;
using VoiceBox.Assistant.Runtime.TurnContext;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class ControlLoopContractTests
{
    [Fact]
    public void ControlPlane_ExposesClosedTypedIngressLoopPublicationAndEffectBoundaries()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;

        Type runtimeCommand = Require(assembly, "VoiceBox.Assistant.Runtime.Control.RuntimeCommand");
        Type assistantSignal = Require(assembly, "VoiceBox.Assistant.Runtime.Control.AssistantSignal");
        Type envelope = Require(assembly, "VoiceBox.Assistant.Runtime.Control.ControlEnvelope");
        Type ingress = Require(assembly, "VoiceBox.Assistant.Runtime.Control.AssistantControlIngress");
        Type loop = Require(assembly, "VoiceBox.Assistant.Runtime.Control.AssistantControlLoop");
        Type publication = Require(assembly, "VoiceBox.Assistant.Runtime.Publications.OrderedPublicationBoundary");
        Type effect = Require(assembly, "VoiceBox.Assistant.Kernel.Effects.AssistantEffect");
        Type effectRunner = Require(assembly, "VoiceBox.Assistant.Runtime.Effects.AssistantEffectRunner");

        Assert.True(runtimeCommand.IsValueType || runtimeCommand.IsAbstract);
        Assert.True(assistantSignal.IsValueType || assistantSignal.IsAbstract);
        Assert.True(envelope.IsValueType);
        Assert.NotNull(envelope.GetProperty("IngressSequence"));
        Assert.NotNull(envelope.GetProperty("SourceTimestamp"));
        Assert.NotNull(envelope.GetProperty("Source"));
        Assert.NotNull(ingress.GetProperty("CriticalCapacity"));
        Assert.NotNull(ingress.GetProperty("CompletionAndLifecycleCapacity"));
        Assert.NotNull(ingress.GetMethod("TryEnqueueCritical"));
        Assert.NotNull(ingress.GetMethod("EnqueueAsync"));
        Assert.NotNull(loop.GetMethod("Start"));
        Assert.NotNull(publication.GetProperty("NextPublicationSequence"));
        Assert.True(effect.IsAbstract);
        Assert.NotNull(effectRunner.GetMethod("ExecuteAsync"));
        Assert.Null(assembly.GetType("VoiceBox.Assistant.State.AssistantTurnControlPorts"));
    }

    [Fact]
    public void CapabilityModules_DoNotOwnTheTurnKernelOrPublicEventDispatcher()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type[] turnOwners =
        [
            Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.SpeechCandidateOwner"),
            Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.BargeInValidationTurnOwner"),
            Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionTurnOwner"),
            Require(assembly, "VoiceBox.Assistant.Runtime.Control.AssistantControlLoopTurnWriter"),
        ];
        Type eventDispatcher = Require(assembly, "VoiceBox.Assistant.Runtime.Publications.AssistantEventDispatcher");

        foreach (Type module in new[]
        {
            typeof(RecognitionModule),
            typeof(ResponseModule),
            typeof(ActiveNearEndCoordinator),
            typeof(BargeInRecognitionOutcomeCoordinator),
        })
        {
            FieldInfo[] fields = module.GetFields(
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            Assert.DoesNotContain(
                fields,
                field => Array.Exists(turnOwners, owner => field.FieldType == owner));
            Assert.DoesNotContain(fields, field => field.FieldType == eventDispatcher);
        }
    }

    [Fact]
    public void CapabilityCoordinators_ReturnSemanticPlansWithoutOwningKernelCommitCapability()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Assert.Null(assembly.GetType("VoiceBox.Assistant.Runtime.Control.ITurnSemanticRequestPort"));
        Assert.Null(assembly.GetType("VoiceBox.Assistant.Runtime.Control.TurnSemanticRequestPort"));

        Type commitPort = Require(
            assembly,
            "VoiceBox.Assistant.Runtime.Control.ITurnSemanticCommitPort");
        Type writer = Require(
            assembly,
            "VoiceBox.Assistant.Runtime.Control.AssistantControlLoopTurnWriter");
        Type router = Require(
            assembly,
            "VoiceBox.Assistant.Runtime.Control.AssistantControlLoopRouter");
        Type[] featureCoordinators =
        [
            typeof(SpeechBoundaryCoordinator),
            typeof(BargeInTurnCoordinator),
            typeof(RecognitionTurnCoordinator),
            typeof(WakeupInterruptOwner),
        ];

        foreach (Type feature in featureCoordinators)
        {
            FieldInfo[] fields = feature.GetFields(
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            Assert.DoesNotContain(fields, field => field.FieldType == commitPort || field.FieldType == writer);
        }

        Assert.Contains(
            router.GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            field => field.FieldType == commitPort);
        Assert.Equal(
            ["Commit"],
            commitPort.GetMethods().Select(static method => method.Name).ToArray());
    }

    [Fact]
    public void Runtime_OwnsExactlyOneControlLoopWithoutDynamicRoutingSurfaces()
    {
        Assert.NotNull(Require(
            typeof(VoiceAssistant).Assembly,
            "VoiceBox.Assistant.Runtime.Control.AssistantControlLoop"));

        // Collaborators may share a reference to the loop and the effect runner, so "exactly one"
        // is a construction fact: the whole assembly creates each of them in a single place.
        Assert.Equal(1, CountConstructions("AssistantControlLoop"));
        Assert.Equal(1, CountConstructions("AssistantEffectRunner"));

        FieldInfo[] fields = ShellSurface.Fields;
        Assert.DoesNotContain(fields, static field =>
            field.FieldType == typeof(IServiceProvider) ||
            field.FieldType.Name.Contains("ServiceLocator", StringComparison.Ordinal) ||
            field.FieldType.Name.Contains("Plugin", StringComparison.Ordinal));
    }

    [Fact]
    public void SupersededObservationStoreAndCoordinator_AreReplacedByTypedTurnOwners()
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Assert.Null(assembly.GetType("VoiceBox.Assistant.State.AssistantTurnObservationStore"));
        Assert.Null(assembly.GetType("VoiceBox.Assistant.Runtime.AssistantTurnObservationCoordinator"));

        Type candidates = Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.SpeechCandidateOwner");
        Type validation = Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.BargeInValidationTurnOwner");
        Type recognitionTurn = Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionTurnOwner");

        // The facade holds none of the three owners: it can no longer observe or mutate a turn.
        FieldInfo[] runtimeFields = typeof(AssistantRuntime).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.DeclaredOnly);
        Assert.DoesNotContain(
            runtimeFields,
            field => field.FieldType == candidates ||
                field.FieldType == validation ||
                field.FieldType == recognitionTurn);

        // Every interruption outcome still observes before it commits, now on its owner.
        AssertCalls(
            typeof(ActiveNearEndControlHandler),
            "ProcessInterruptionRequested",
            typeof(BargeInTurnCoordinator),
            "TryConfirmFromNearEndWindow");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "CommitInterruptionRequested",
            typeof(ConfirmedBargeInApplier),
            "Apply");
        AssertCalls(
            typeof(WakeupInterruptOwner),
            "ObserveInterrupt",
            typeof(TurnCompletionCoordinator),
            "TryObserveWakeupInterrupt");
        AssertCalls(
            typeof(WakeupInterruptOwner),
            "CommitInterrupt",
            typeof(TurnCompletionCoordinator),
            "CommitWakeupInterrupt");
        AssertCalls(
            typeof(RecognitionTurnCoordinator),
            "ObserveResetAfterAsrStartFailure",
            typeof(IBargeInTurnSupportPort),
            "TryObserveAsrStartFailure");
        AssertCalls(
            typeof(RecognitionTurnCoordinator),
            "CommitResetAfterAsrStartFailure",
            typeof(IBargeInTurnSupportPort),
            "CommitAsrStartFailure");
        AssertDoesNotCall(
            typeof(RecognitionTurnCoordinator),
            "ObserveResetAfterAsrStartFailure",
            typeof(INearEndCaptureControlPort),
            "StopCapture");
        AssertCalls(
            typeof(RecognitionTurnCoordinator),
            "CommitResetAfterAsrStartFailure",
            typeof(INearEndCaptureControlPort),
            "StopCapture");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "ObserveValidatedBargeIn",
            validation,
            "TryObserveValidated");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "CommitValidationPlan",
            validation,
            "CommitAccepted");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "ObserveRejectedBargeIn",
            validation,
            "TryObserveRejection");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "CommitValidationPlan",
            validation,
            "CommitRejected");
        AssertDoesNotCall(
            typeof(BargeInTurnCoordinator),
            "ObserveConfirmedBargeIn",
            typeof(BargeInDiagnosticsCoordinator),
            "RecordConfirmed");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "CommitInterruptionRequested",
            typeof(BargeInDiagnosticsCoordinator),
            "RecordConfirmed");
        AssertDoesNotCall(
            typeof(BargeInTurnCoordinator),
            "ObserveValidatedBargeIn",
            typeof(IConversationRollbackPort),
            "RemoveUnansweredUserMessage");
        AssertDoesNotCall(
            typeof(BargeInTurnCoordinator),
            "ObserveValidatedBargeIn",
            typeof(TurnContextStore),
            "TryCreate");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "CommitValidationPlan",
            typeof(IConversationRollbackPort),
            "RemoveUnansweredUserMessage");
        AssertCalls(
            typeof(BargeInTurnCoordinator),
            "CommitValidationPlan",
            typeof(TurnContextStore),
            "TryCreate");
        AssertDoesNotCall(
            typeof(RecognitionTurnCoordinator),
            "ObserveSessionStarted",
            typeof(TurnContextStore),
            "TryAttachAsrSession");
        AssertCalls(
            typeof(RecognitionTurnCoordinator),
            "CommitSessionStarted",
            typeof(TurnContextStore),
            "TryAttachAsrSession");
        AssertCalls(
            typeof(RecognitionTurnCoordinator),
            "CommitSemanticPlan",
            typeof(RecognitionTurnOwner),
            "CommitEnterRecognizing");
    }

    [Fact]
    public void ResponseModule_UsesTypedControlSignalsAndOnlyTheAssistantTextStreamEndpoint()
    {
        FieldInfo[] fields = typeof(ResponseModule).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        string[] fieldTypes = fields
            .Select(static field => field.FieldType.FullName ?? field.FieldType.Name)
            .ToArray();

        Assert.Contains("VoiceBox.Assistant.Runtime.Control.IResponseControlSignalPort", fieldTypes);
        Assert.Contains("VoiceBox.Assistant.Runtime.Publications.IAssistantTextStreamSink", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Assistant.State.IResponseTurnControlPort", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Assistant.Runtime.Publications.IAssistantPublicEventSink", fieldTypes);
    }

    [Fact]
    public void RecognitionModule_ReportsTypedCompletionWithoutOwningTurnOrPublicationPorts()
    {
        FieldInfo[] fields = typeof(RecognitionModule).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        string[] fieldTypes = fields
            .Select(static field => field.FieldType.FullName ?? field.FieldType.Name)
            .ToArray();

        Assert.Contains("VoiceBox.Assistant.Runtime.Control.IRecognitionControlSignalPort", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Assistant.State.IRecognitionTurnControlPort", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Assistant.Runtime.Publications.IAssistantPublicEventSink", fieldTypes);
    }

    [Fact]
    public void ValidationModule_ReportsTypedOutcomesWithoutOwningTurnOrPublicationPorts()
    {
        FieldInfo[] fields = typeof(BargeInRecognitionOutcomeCoordinator).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        string[] fieldTypes = fields
            .Select(static field => field.FieldType.FullName ?? field.FieldType.Name)
            .ToArray();

        Assert.Contains("VoiceBox.Assistant.Runtime.Control.IBargeInValidationControlSignalPort", fieldTypes);
        Assert.DoesNotContain(typeof(ResponseModule).FullName!, fieldTypes);
        Assert.DoesNotContain("VoiceBox.Assistant.State.IBargeInValidationTurnControlPort", fieldTypes);
        Assert.DoesNotContain("VoiceBox.Assistant.Runtime.Publications.IAssistantPublicEventSink", fieldTypes);
    }

    [Fact]
    public void KernelResponseAsyncStateMachine_DoesNotDirectlyMutateOrReadTheTurnStateMachine()
    {
        // The response effect now lives in its own typed capability handler; it must still be
        // unable to reach the turn observation store directly.
        MethodInfo pipeline = typeof(ResponseEffectHandler).GetMethod(
            "RunAsync",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException("ResponseEffectHandler.RunAsync was not found.");
        var stateMachine = pipeline.GetCustomAttribute<AsyncStateMachineAttribute>()?.StateMachineType
            ?? throw new Xunit.Sdk.XunitException("ResponseEffectHandler.RunAsync must be an async state machine.");
        MethodInfo moveNext = stateMachine.GetMethod(
            "MoveNext",
            BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException("ResponseEffectHandler.RunAsync state machine MoveNext was not found.");

        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type[] owners =
        [
            Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.SpeechCandidateOwner"),
            Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.BargeInValidationTurnOwner"),
            Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionTurnOwner"),
        ];
        Assert.DoesNotContain(
            GetCalledMethods(moveNext),
            method => Array.Exists(owners, owner => method.DeclaringType == owner));
    }

    [Fact]
    public void AssistantRuntime_DoesNotImplementAnyEffectHandlerFamily()
    {
        // The shell may schedule and own resources, but each capability family must be answered
        // by its own handler type (architecture §5#4).
        Assert.DoesNotContain(
            typeof(AssistantRuntime).GetInterfaces(),
            type => type == typeof(IInputGateEffectHandler) ||
                type == typeof(IRecognitionEffectHandler) ||
                type == typeof(IResponseEffectHandler) ||
                type == typeof(IPlaybackEffectHandler) ||
                type == typeof(IConversationEffectHandler) ||
                type == typeof(IRuntimeEffectHandler) ||
                type == typeof(IPublicEventEffectHandler));
    }

    [Theory]
    [InlineData("OnSpeechStarted")]
    [InlineData("OnSpeaking")]
    [InlineData("OnSpeechEnded")]
    public void VadCallbacks_OnlyCaptureAndEnqueueFactsWithoutDirectTurnStateMachineAccess(
        string callbackName)
    {
        // The callbacks moved off the facade onto the turn-boundary ingress, and still only
        // normalise, stamp and admit: none of them reaches a turn-observation owner.
        Assert.Null(typeof(AssistantRuntime).GetMethod(
            callbackName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic));
        AssertOwnsNoTurnObservation(typeof(SpeechBoundaryIngress), callbackName);
    }

    /// <summary>
    /// Counts every construction of a type across production sources, covering both the explicit
    /// and the target-typed <c>new</c> form.
    /// </summary>
    private static int CountConstructions(string typeName)
    {
        var pattern = new System.Text.RegularExpressions.Regex(
            $@"(new\s+{typeName}\s*\()|({typeName}\s+\w+\s*=\s*new\s*\()");
        string root = Path.Combine(RepositoryRoot(), "src", "VoiceBox.Assistant");
        return Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories)
            .Sum(file => pattern.Matches(File.ReadAllText(file)).Count);
    }

    private static string RepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !Directory.Exists(Path.Combine(directory.FullName, "src")))
            directory = directory.Parent;

        return directory?.FullName
            ?? throw new Xunit.Sdk.XunitException("Repository root was not found.");
    }

    private static Type Require(Assembly assembly, string name)
        => assembly.GetType(name)
            ?? throw new Xunit.Sdk.XunitException($"Required control-plane contract type '{name}' was not found.");

    private static void AssertCalls(Type ownerType, string ownerMethodName, Type calleeType, string calleeMethodName)
    {
        MethodInfo ownerMethod = ownerType.GetMethod(
            ownerMethodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException(
                $"{ownerType.Name} method '{ownerMethodName}' was not found.");

        Assert.Contains(
            GetCalledMethods(ownerMethod),
            method => method.DeclaringType == calleeType && method.Name == calleeMethodName);
    }

    private static void AssertDoesNotCall(
        Type ownerType,
        string ownerMethodName,
        Type calleeType,
        string calleeMethodName)
    {
        MethodInfo ownerMethod = ownerType.GetMethod(
            ownerMethodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException(
                $"{ownerType.Name} method '{ownerMethodName}' was not found.");

        Assert.DoesNotContain(
            GetCalledMethods(ownerMethod),
            method => method.DeclaringType == calleeType && method.Name == calleeMethodName);
    }

    private static void AssertOwnsNoTurnObservation(Type type, string methodName)
    {
        Assembly assembly = typeof(VoiceAssistant).Assembly;
        Type[] owners =
        [
            Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.SpeechCandidateOwner"),
            Require(assembly, "VoiceBox.Assistant.Modules.BargeIn.BargeInValidationTurnOwner"),
            Require(assembly, "VoiceBox.Assistant.Modules.Recognition.Controllers.RecognitionTurnOwner"),
        ];
        MethodInfo method = type.GetMethod(
            methodName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException($"{type.Name}.{methodName} was not found.");

        Assert.DoesNotContain(
            GetCalledMethods(method),
            called => Array.Exists(owners, owner => called.DeclaringType == owner));
    }

    private static IEnumerable<MethodBase> GetCalledMethods(MethodInfo method)
    {
        byte[] il = method.GetMethodBody()?.GetILAsByteArray()
            ?? throw new Xunit.Sdk.XunitException($"Method '{method.Name}' has no IL body.");
        Type[]? typeArguments = method.DeclaringType?.GetGenericArguments();
        Type[]? methodArguments = method.GetGenericArguments();
        int offset = 0;

        while (offset < il.Length)
        {
            OpCode opcode = ReadOpCode(il, ref offset);
            if (opcode.OperandType == OperandType.InlineMethod)
            {
                int token = BitConverter.ToInt32(il, offset);
                offset += sizeof(int);
                yield return method.Module.ResolveMethod(token, typeArguments, methodArguments)
                    ?? throw new Xunit.Sdk.XunitException(
                        $"Unable to resolve IL method token 0x{token:X8} in '{method.Name}'.");
                continue;
            }

            offset += OperandSize(opcode.OperandType, il, offset);
        }
    }

    private static OpCode ReadOpCode(byte[] il, ref int offset)
    {
        short value = il[offset++];
        if (value == 0xFE)
            value = (short)(0xFE00 | il[offset++]);

        return OpCodesByValue.TryGetValue(value, out OpCode opcode)
            ? opcode
            : throw new Xunit.Sdk.XunitException($"Unknown IL opcode 0x{value:X4}.");
    }

    private static int OperandSize(OperandType operandType, byte[] il, int offset)
        => operandType switch
        {
            OperandType.InlineNone => 0,
            OperandType.ShortInlineBrTarget or OperandType.ShortInlineI or OperandType.ShortInlineVar => 1,
            OperandType.InlineVar => 2,
            OperandType.InlineBrTarget or OperandType.InlineField or OperandType.InlineI or
                OperandType.InlineSig or OperandType.InlineString or OperandType.InlineTok or
                OperandType.InlineType => 4,
            OperandType.InlineI8 or OperandType.InlineR => 8,
            OperandType.ShortInlineR => 4,
            OperandType.InlineSwitch => sizeof(int) + BitConverter.ToInt32(il, offset) * sizeof(int),
            _ => throw new Xunit.Sdk.XunitException($"Unsupported IL operand type '{operandType}'."),
        };

    private static readonly IReadOnlyDictionary<short, OpCode> OpCodesByValue = typeof(OpCodes)
        .GetFields(BindingFlags.Public | BindingFlags.Static)
        .Where(static field => field.FieldType == typeof(OpCode))
        .Select(static field => (OpCode)field.GetValue(null)!)
        .ToDictionary(static opcode => opcode.Value);
}
