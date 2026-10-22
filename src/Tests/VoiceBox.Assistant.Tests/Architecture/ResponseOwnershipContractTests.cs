using System.Reflection;
using System.Reflection.Emit;
using VoiceBox.Assistant.Composition.Wiring;
using VoiceBox.Assistant.Modules.Response.Controllers;
using VoiceBox.Assistant.Runtime.Control.Contracts;
using VoiceBox.Assistant.Runtime.Effects.Contracts;
using Xunit;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// <c>ResponseModule</c> consumes the generate/cancel response effects and uniquely owns the
/// current response operation, as every module owns its own session and operation state. These
/// tests prove exclusive create / cancel / complete / fail ownership. They do not assert that
/// <c>AssistantRuntime</c> stays off <see cref="IResponseControlSignalPort"/>; that is covered by
/// the control-loop contracts.
/// </summary>
public sealed class ResponseOwnershipContractTests
{
    [Fact]
    public void ResponseModule_MustOwnResponseOperationLifecycle()
    {
        Type owner = typeof(ResponseModule);
        Assert.True(owner.IsSealed, "ResponseModule must be the sealed response-operation owner.");

        Type operation = owner.GetNestedType("ResponseOperation", BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException(
                "ResponseModule must declare a private ResponseOperation that owns the CTS.");
        Assert.Contains(
            operation.GetProperties(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic),
            property => property.PropertyType == typeof(CancellationTokenSource));
        Assert.Contains(
            owner.GetFields(BindingFlags.Instance | BindingFlags.NonPublic),
            field => field.FieldType == operation);

        MethodInfo cancel = owner.GetMethod(
            "Cancel",
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException(
                "ResponseModule must expose Cancel(turnId, operationId) as the exclusive cancel path.");
        Assert.Equal(typeof(bool), cancel.ReturnType);
        Assert.Equal(new[] { typeof(long), typeof(long) }, cancel.GetParameters().Select(p => p.ParameterType));
        Assert.NotNull(owner.GetMethod(
            "BeginOperation",
            BindingFlags.Instance | BindingFlags.NonPublic));

        HashSet<MethodBase> ownerCalls = CollectCalls(owner);
        Assert.Contains(
            ownerCalls,
            method => method.Name == nameof(IResponseControlSignalPort.PublishResponseCompletedAsync));
        Assert.Contains(
            ownerCalls,
            method => method.Name == nameof(IResponseControlSignalPort.PublishResponseTerminatedAsync));
        Assert.Contains(
            ownerCalls,
            method => method.DeclaringType == typeof(CancellationTokenSource) &&
                method.Name is "Cancel" or "CancelAsync");

        var violations = new List<string>();
        Assembly assembly = typeof(VoiceAssistant).Assembly;

        foreach (Type implementor in assembly.GetTypes().Where(static type =>
            type.IsClass &&
            !type.IsAbstract &&
            typeof(IResponseEffectHandler).IsAssignableFrom(type) &&
            type != typeof(ResponseModule)))
        {
            if (IsForbiddenLifecycleProxyName(implementor.Name))
            {
                violations.Add(
                    $"{implementor.FullName} is a Coordinator/Manager/Orchestrator/Proxy/Facade " +
                    "standing in for ResponseModule operation ownership.");
            }

            foreach (FieldInfo field in implementor.GetFields(
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (field.FieldType == typeof(IActiveResponseCancellationPort))
                {
                    violations.Add(
                        $"{implementor.FullName} holds {nameof(IActiveResponseCancellationPort)}; " +
                        "cancel of the response operation must be exclusive to ResponseModule.Cancel.");
                }
            }

            foreach (MethodBase method in DeclaredMethods(implementor))
            {
                foreach (MethodBase callee in GetCalledMethods(method))
                {
                    if (IsCancellationTokenSourceConstruction(callee))
                    {
                        violations.Add(
                            $"{Format(method)} constructs CancellationTokenSource; " +
                            "create of the response-operation CTS must be exclusive to ResponseModule.");
                    }

                    if (callee.Name is "CancelActiveResponse" or "CancelResponseCts")
                    {
                        violations.Add(
                            $"{Format(method)} calls {callee.DeclaringType?.Name}.{callee.Name}; " +
                            "cancel of the response operation must be exclusive to ResponseModule.Cancel.");
                    }

                    if (callee.DeclaringType == typeof(CancellationTokenSource) &&
                        callee.Name is "Cancel" or "CancelAsync")
                    {
                        violations.Add(
                            $"{Format(method)} cancels a CancellationTokenSource; " +
                            "cancel of the response operation must be exclusive to ResponseModule.Cancel.");
                    }
                }
            }
        }

        foreach (Type type in assembly.GetTypes())
        {
            if (type.IsInterface || IsOwnedBy(type, owner))
                continue;
            if (typeof(IResponseControlSignalPort).IsAssignableFrom(type))
                continue;

            foreach (MethodBase method in DeclaredMethods(type))
            {
                foreach (MethodBase callee in GetCalledMethods(method))
                {
                    if (callee.Name is
                            nameof(IResponseControlSignalPort.PublishResponseCompletedAsync) or
                            nameof(IResponseControlSignalPort.PublishResponseTerminatedAsync) &&
                        callee.DeclaringType == typeof(IResponseControlSignalPort))
                    {
                        violations.Add(
                            $"{Format(method)} publishes response complete/fail; " +
                            "complete and fail must be exclusive to ResponseModule.");
                    }
                }
            }
        }

        foreach (Type resourceOwner in assembly.GetTypes().Where(static type =>
            type.IsClass && typeof(IResponseTurnResourcePort).IsAssignableFrom(type)))
        {
            foreach (MethodBase method in DeclaredMethods(resourceOwner)
                .Where(static method => method.Name is "TryAttachResponseContext" or "TryAttachResponseTurnContext"))
            {
                if (GetCalledMethods(method).Any(IsCancellationTokenSourceConstruction))
                {
                    violations.Add(
                        $"{Format(method)} constructs CancellationTokenSource; " +
                        "attach may correlate an already-created operation but must not create it.");
                }
            }
        }

        Assert.True(
            violations.Count == 0,
            "ResponseModule does not exclusively own response operation lifecycle " +
            "(create, cancel, complete, fail):" + Environment.NewLine +
            string.Join(Environment.NewLine, violations.Distinct()));
    }

    private static bool IsForbiddenLifecycleProxyName(string name)
        => name.Contains("Coordinator", StringComparison.Ordinal) ||
            name.Contains("Manager", StringComparison.Ordinal) ||
            name.Contains("Orchestrator", StringComparison.Ordinal) ||
            name.Contains("Proxy", StringComparison.Ordinal) ||
            name.Contains("Facade", StringComparison.Ordinal);

    private static bool IsOwnedBy(Type type, Type owner)
    {
        for (Type? current = type; current is not null; current = current.DeclaringType)
        {
            if (current == owner)
                return true;
        }

        return false;
    }

    private static bool IsCancellationTokenSourceConstruction(MethodBase method)
        => method.DeclaringType == typeof(CancellationTokenSource) &&
            (method.IsConstructor || method.Name == nameof(CancellationTokenSource.CreateLinkedTokenSource));

    private static HashSet<MethodBase> CollectCalls(Type type)
    {
        var calls = new HashSet<MethodBase>();
        foreach (MethodBase method in DeclaredMethods(type))
            calls.UnionWith(GetCalledMethods(method));
        return calls;
    }

    private static IEnumerable<MethodBase> DeclaredMethods(Type type)
    {
        const BindingFlags declared =
            BindingFlags.Instance | BindingFlags.Static | BindingFlags.Public |
            BindingFlags.NonPublic | BindingFlags.DeclaredOnly;

        foreach (Type current in SelfAndNested(type))
        {
            foreach (MethodInfo method in current.GetMethods(declared))
                yield return method;
            foreach (ConstructorInfo constructor in current.GetConstructors(declared))
                yield return constructor;
        }
    }

    private static IEnumerable<Type> SelfAndNested(Type type)
    {
        yield return type;
        foreach (Type nested in type.GetNestedTypes(BindingFlags.Public | BindingFlags.NonPublic))
        {
            foreach (Type child in SelfAndNested(nested))
                yield return child;
        }
    }

    private static IEnumerable<MethodBase> GetCalledMethods(MethodBase method)
    {
        byte[]? il = method.GetMethodBody()?.GetILAsByteArray();
        if (il is null)
            yield break;

        Type[]? typeArguments = method.DeclaringType?.GetGenericArguments();
        Type[]? methodArguments = method is MethodInfo methodInfo ? methodInfo.GetGenericArguments() : [];
        int offset = 0;
        while (offset < il.Length)
        {
            OpCode opcode = ReadOpCode(il, ref offset);
            if (opcode.OperandType is OperandType.InlineMethod or OperandType.InlineTok)
            {
                int token = BitConverter.ToInt32(il, offset);
                offset += sizeof(int);
                MethodBase? resolved = null;
                try
                {
                    resolved = method.Module.ResolveMethod(token, typeArguments, methodArguments);
                }
                catch (ArgumentException)
                {
                }

                if (resolved is not null)
                    yield return resolved;
                continue;
            }

            offset += OperandSize(opcode.OperandType, il, offset);
        }
    }

    private static OpCode ReadOpCode(byte[] il, ref int offset)
    {
        short value = il[offset++];
        if (value == 0xFE)
            value = (short)((0xFE << 8) | il[offset++]);

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
                OperandType.InlineSig or OperandType.InlineString or OperandType.InlineType => 4,
            OperandType.InlineI8 or OperandType.InlineR => 8,
            OperandType.ShortInlineR => 4,
            OperandType.InlineSwitch => sizeof(int) + BitConverter.ToInt32(il, offset) * sizeof(int),
            _ => throw new Xunit.Sdk.XunitException($"Unsupported IL operand type '{operandType}'."),
        };

    private static string Format(MethodBase method)
        => $"{method.DeclaringType?.FullName}.{method.Name}";

    private static readonly IReadOnlyDictionary<short, OpCode> OpCodesByValue = typeof(OpCodes)
        .GetFields(BindingFlags.Public | BindingFlags.Static)
        .Where(static field => field.FieldType == typeof(OpCode))
        .Select(static field => (OpCode)field.GetValue(null)!)
        .ToDictionary(static opcode => opcode.Value);
}
