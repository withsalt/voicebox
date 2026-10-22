using Microsoft.ML.OnnxRuntime;
using System.Reflection;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Core.Onnx;
using VoiceBox.DTD.RenderAware.Models;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Hosting;
using VoiceBox.Assistant.Runtime;
using Xunit;
using KernelEffectKind = VoiceBox.Assistant.Kernel.Effects.AssistantEffectKind;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    /// <summary>
    /// Hosting is a source-level boundary inside the core assembly, not a separate project. See
    /// <c>docs/voicebox-assistant-architecture.md</c> §7.1.
    /// </summary>
    [Fact]
    public void HostingIsCoLocatedInTheCoreAssembly()
    {
        Assembly assistant = typeof(VoiceAssistant).Assembly;
        Assembly hosting = typeof(VoiceAssistantServiceCollectionExtensions).Assembly;
        Assert.Equal(assistant, hosting);
    }

    /// <summary>
    /// Kernel, Runtime and Modules share one assembly because they share internal typed plans and
    /// have no independent dependency or release value. Splitting them for directory symmetry is a
    /// deliberate non-goal; see <c>docs/voicebox-assistant-architecture.md</c> §7.1.
    /// </summary>
    [Fact]
    public void LogicalBoundariesAreNotSplitIntoSeparateProjects()
    {
        string sourceRoot = Path.Combine(FindRepositoryRoot(), "src");
        foreach (string project in new[]
        {
            "VoiceBox.Assistant.Abstractions",
            "VoiceBox.Assistant.Kernel",
            "VoiceBox.Assistant.Runtime",
            "VoiceBox.Assistant.Modules",
            "VoiceBox.Assistant.Integrations.RenderAware",
        })
        {
            Assert.False(Directory.Exists(Path.Combine(sourceRoot, project)),
                $"'{project}' would be an assembly split with no independent dependency value.");
        }
    }

    [Fact]
    public void KernelSourceDoesNotDependOnRuntimeCompositionOrProviders()
    {
        string kernelRoot = Path.Combine(
            FindRepositoryRoot(),
            "src",
            "VoiceBox.Assistant",
            "Kernel");
        (string Marker, string Boundary)[] forbidden =
        [
            ("VoiceBox.Assistant.Runtime", "Runtime"),
            ("VoiceBox.Assistant.Composition", "Composition"),
            ("VoiceBox.Assistant.Modules", "capability modules"),
            ("VoiceBox.Core.Services", "VoiceBox providers"),
            ("Microsoft.Extensions.Logging", "logging"),
            ("System.IO", "I/O"),
            ("System.Threading", "concurrency"),
            ("System.Collections.Concurrent", "concurrency"),
            ("ModelContextProtocol", "MCP SDK/wire types"),
            ("IServiceProvider", "service location"),
            ("AssistantCapabilityGraph", "capability lookup"),
            ("Task", "tasks"),
            ("CancellationToken", "cancellation resources"),
            ("TimeProvider", "clocks"),
            ("Random", "randomness"),
            ("Queue<", "queues"),
            ("Channel<", "queues"),
        ];

        string[] violations = EnumerateSourceFiles(kernelRoot)
            .SelectMany(path =>
            {
                string source = File.ReadAllText(path);
                return forbidden
                    .Where(rule => source.Contains(rule.Marker, StringComparison.Ordinal))
                    .Select(rule =>
                        $"{Path.GetRelativePath(kernelRoot, path)} references {rule.Boundary} via '{rule.Marker}'.");
            })
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.True(
            violations.Length == 0,
            "Kernel dependency violations:" + Environment.NewLine + string.Join(Environment.NewLine, violations));
    }

    [Fact]
    public void RuntimeUsesOnlyTypedPlansAndHasNoLocatorOrGraphOwnership()
    {
        FieldInfo[] fields = typeof(AssistantRuntime).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        Assert.DoesNotContain(fields, field =>
            field.FieldType == typeof(IServiceProvider) ||
            field.FieldType.Name is "IAudioServiceManager" or "AssistantCapabilityGraph" ||
            field.Name.Contains("serviceProvider", StringComparison.OrdinalIgnoreCase) ||
            field.Name.Contains("capabilityGraph", StringComparison.OrdinalIgnoreCase) ||
            field.Name.Contains("registry", StringComparison.OrdinalIgnoreCase));

        string source = ReadSource("src", "VoiceBox.Assistant", "Runtime", "AssistantRuntime.cs");
        Assert.DoesNotContain("IServiceProvider", source, StringComparison.Ordinal);
        Assert.DoesNotContain("IAudioServiceManager", source, StringComparison.Ordinal);
        Assert.DoesNotContain("AssistantCapabilityGraph", source, StringComparison.Ordinal);
        Assert.DoesNotContain("GetService", source, StringComparison.Ordinal);
    }

    [Fact]
    public void AssistantDependsOnAcousticDelayContract_NotWebRtcProvider()
    {
        AssemblyName[] references = typeof(VoiceAssistant).Assembly.GetReferencedAssemblies();
        Assert.DoesNotContain(
            references,
            reference => string.Equals(
                reference.Name,
                "VoiceBox.APM.WebRtc",
                StringComparison.Ordinal));

        string project = ReadSource(
            "src",
            "VoiceBox.Assistant",
            "VoiceBox.Assistant.csproj");
        Assert.DoesNotContain("VoiceBox.APM.WebRtc", project, StringComparison.Ordinal);

        string builder = ReadSource(
            "src",
            "VoiceBox.Assistant",
            "Composition",
            "Builder",
            "VoiceAssistantBuilder.cs");
        Assert.Contains("IAcousticPathDelayEstimatorFactory", builder, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimePlanCarriesOnlyModulePortsAndAdapterFactories()
    {
        FieldInfo[] fields = typeof(AssistantRuntimePlan).GetFields(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        PropertyInfo[] properties = typeof(AssistantRuntimePlan).GetProperties(
            BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);

        Assert.DoesNotContain(
            fields.Select(static field => field.FieldType)
                .Concat(properties.Select(static property => property.PropertyType)),
            type => type.Namespace?.StartsWith("VoiceBox.Core.Services.", StringComparison.Ordinal) == true);

        string plan = ReadSource("src", "VoiceBox.Assistant", "Composition", "Plans", "AssistantRuntimePlan.cs");
        Assert.DoesNotContain("VoiceBox.Core.Services", plan, StringComparison.Ordinal);
        Assert.Contains("IRecognitionProviderAdapter", plan, StringComparison.Ordinal);
        Assert.Contains("IResponseGeneratorFactory", plan, StringComparison.Ordinal);
        Assert.Contains("ISpeechSynthesisStreamPort", plan, StringComparison.Ordinal);
    }

    [Fact]
    public void AssistantCompositionHasNoServiceManagerResolution()
    {
        string assistantRoot = Path.Combine(FindRepositoryRoot(), "src", "VoiceBox.Assistant");
        Assert.DoesNotContain(
            EnumerateSourceFiles(assistantRoot),
            path => File.ReadAllText(path).Contains("IAudioServiceManager", StringComparison.Ordinal));

    }

    [Fact]
    public void ServiceProviderAppearsOnlyAtHostingOuterCompositionFactory()
    {
        string assistantRoot = Path.Combine(FindRepositoryRoot(), "src", "VoiceBox.Assistant");
        string[] coreSources = EnumerateSourceFiles(assistantRoot).ToArray();
        Assert.DoesNotContain(coreSources, path =>
            !path.Contains("\\Hosting\\", StringComparison.OrdinalIgnoreCase) &&
            File.ReadAllText(path).Contains("IServiceProvider", StringComparison.Ordinal));

        string hostingRoot = Path.Combine(assistantRoot, "Hosting");
        string[] hostingSources = EnumerateSourceFiles(hostingRoot).ToArray();
        Assert.All(hostingSources, path =>
        {
            string source = File.ReadAllText(path);
            if (source.Contains("IServiceProvider", StringComparison.Ordinal))
            {
                Assert.EndsWith("VoiceAssistantServiceCollectionExtensions.cs", path,
                    StringComparison.OrdinalIgnoreCase);
                Assert.Contains("CreateHostInstance(IServiceProvider provider)", source, StringComparison.Ordinal);
                Assert.Contains("CreateBuilder(IServiceProvider provider)", source, StringComparison.Ordinal);
            }
        });
    }

    [Fact]
    public void CoreHasClosedEffectDispatchAndNoDynamicPluginOrHotSwapSurface()
    {
        string assistantRoot = Path.Combine(FindRepositoryRoot(), "src", "VoiceBox.Assistant");
        string source = string.Join('\n', EnumerateSourceFiles(assistantRoot)
            .Where(path => !path.Contains("\\Hosting\\", StringComparison.OrdinalIgnoreCase))
            .Select(File.ReadAllText));

        Assert.DoesNotContain("VoiceAssistantKernel", source, StringComparison.Ordinal);
        Assert.DoesNotContain("AssistantContext", source, StringComparison.Ordinal);
        Assert.DoesNotContain("IServiceProvider", source, StringComparison.Ordinal);
        Assert.DoesNotContain("GetService", source, StringComparison.Ordinal);
        Assert.DoesNotContain("Get<", source, StringComparison.Ordinal);
        Assert.DoesNotContain("EnableMcp", source, StringComparison.Ordinal);
        Assert.DoesNotContain("ModelContextProtocol.", source, StringComparison.Ordinal);
        Assert.DoesNotContain("HotSwap", source, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("PluginRegistry", source, StringComparison.Ordinal);
        Assert.DoesNotContain("HandlerRegistry", source, StringComparison.Ordinal);

        string effects = ReadSource("src", "VoiceBox.Assistant", "Runtime", "Effects", "Execution", "AssistantEffectRunner.cs");
        Assert.Contains("return effect switch", effects, StringComparison.Ordinal);
        Assert.Contains("_ => throw new ArgumentOutOfRangeException", effects, StringComparison.Ordinal);
        Assert.DoesNotContain("Dictionary<", effects, StringComparison.Ordinal);
    }

    [Fact]
    public void KernelEffectsUseClosedPayloadTypesAndOwnHistoryCommit()
    {
        string kernel = ReadSource("src", "VoiceBox.Assistant", "Kernel", "TurnKernel.cs");
        string effects = ReadSource("src", "VoiceBox.Assistant", "Runtime", "Effects", "Execution", "AssistantEffectRunner.cs");
        string runtime = ReadSource("src", "VoiceBox.Assistant", "Runtime", "AssistantRuntime.cs");
        string handlers = ReadSource("src", "VoiceBox.Assistant", "Runtime", "Effects", "Handlers", "ConversationEffectHandler.cs");
        string response = ReadSource("src", "VoiceBox.Assistant", "Modules", "Response", "Controllers", "ResponseModule.cs");

        Assert.DoesNotContain("string EventName", kernel, StringComparison.Ordinal);
        Assert.DoesNotContain("TryParseStateTransition", runtime, StringComparison.Ordinal);
        Assert.DoesNotContain("_history.CommitGeneratedResponse", response, StringComparison.Ordinal);
        // History commit belongs to the Conversation effect handler family, not to the shell.
        Assert.DoesNotContain("_history.CommitGeneratedResponse", runtime, StringComparison.Ordinal);
        Assert.Contains("_history.CommitGeneratedResponse(effect.TurnId, effect.Text)", handlers, StringComparison.Ordinal);
        Assert.Contains("PublishStateChangedEffect value =>", effects, StringComparison.Ordinal);
        Assert.Contains("PublishFaultedEffect value =>", effects, StringComparison.Ordinal);
        Assert.Contains("PublishBargeInValidationEffect value =>", effects, StringComparison.Ordinal);
    }

    [Fact]
    public void EffectProtocolHasOneCanonicalKindEnumAndStructuredToolCapabilityIsGraphOnly()
    {
        Assembly assistant = typeof(VoiceAssistant).Assembly;
        Type[] effectKindEnums = assistant.GetTypes()
            .Where(type => type.IsEnum && type.Name == typeof(KernelEffectKind).Name)
            .ToArray();

        Assert.Equal(typeof(KernelEffectKind), Assert.Single(effectKindEnums));
        Assert.Contains(
            "StructuredToolUse",
            Enum.GetNames<AssistantCapability>());
    }

    [Fact]
    public void AddBargeIn_AcceptsACustomOnnxSessionFactoryAndCarriesItIntoTheComposition()
    {
        // AddBargeIn and Build both take a defensive copy of the options. The DTD model settings
        // inherit OnnxExecutionOptions, whose SessionOptionsFactory is a delegate: a copy that
        // cannot carry it either throws out of the public builder or silently reverts a
        // caller-configured ONNX session to the CPU default.
        //
        // The factory records that it was reached and then aborts, because building a real
        // SessionOptions requires the ONNX native library this assembly does not load. Reaching it
        // is exactly the evidence wanted: the delegate survived every clone hop from the caller's
        // options object down to the real detector construction.
        int invocations = 0;
        var reached = new NotSupportedException("session factory reached");
        SessionOptions Factory()
        {
            invocations++;
            throw reached;
        }

        var services = TestServices.Create();
        var options = new VoiceAssistantOptions();
        options.BargeIn.Dtd.Model.SessionOptionsFactory = Factory;
        options.BargeIn.Dtd.Model.ExecutionProvider = OnnxExecutionProvider.Cuda;
        options.BargeIn.Dtd.Model.GpuDeviceId = 2;

        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddBargeIn(options.BargeIn, TestAcousticPathDelayEstimatorFactory.Instance);

        // Before the fix, AddBargeIn's defensive copy threw NotSupportedException with the JSON
        // serializer in the stack, so the public composition path was unusable with a custom
        // ONNX session regardless of whether the model could load.
        AssistantComposition composition = builder.BuildComposition();
        Assert.Contains("BargeIn", composition.Snapshot.Capabilities);
        Assert.Equal(0, invocations);

        // Building the detector must consult the caller's own factory rather than a default.
        Assert.Same(reached, Assert.Throws<NotSupportedException>(() => builder.Build()));
        Assert.Equal(1, invocations);

        // The caller's options object must be left exactly as supplied.
        Assert.Equal<Func<SessionOptions>?>(Factory, options.BargeIn.Dtd.Model.SessionOptionsFactory);
        Assert.Equal(OnnxExecutionProvider.Cuda, options.BargeIn.Dtd.Model.ExecutionProvider);
        Assert.Equal(2, options.BargeIn.Dtd.Model.GpuDeviceId);
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, false, false)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    [InlineData(true, true, true)]
    public void OptionalModuleMatrixUsesOnlyTypedRegistrations(
        bool wakeup,
        bool speaker,
        bool bargeIn)
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantOptions();
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts);
        if (wakeup)
            builder.AddWakeup(services.Wakeup);
        if (speaker)
            builder.AddSpeakerVerification(services.Speaker);
        if (bargeIn)
            builder.AddBargeIn(
                options.BargeIn,
                TestAcousticPathDelayEstimatorFactory.Instance);

        // Online identity is off unless Confirmation.Enable is set; this matrix leaves it at the
        // default (false), so a keyword-only composition is legal and a registered provider
        // installs no speaker module.
        bool identityEffective = false;
        AssistantComposition composition = builder.BuildComposition();
        Assert.Equal(wakeup, composition.Snapshot.Capabilities.Contains("WakeupGate"));
        Assert.Equal(identityEffective, composition.Snapshot.Capabilities.Contains("SpeakerEvidence"));
        Assert.Equal(bargeIn, composition.Snapshot.Capabilities.Contains("BargeIn"));
        // Speaker evidence and the wakeup gate are ready before the first identity frame; barge-in
        // follows them; the audio frontend starts last so no capture frame is lost.
        Assert.Equal(
            Enumerable.Empty<AssistantLifecycleParticipant>()
            .Concat(identityEffective ? [AssistantLifecycleParticipant.SpeakerEvidence] : [])
            .Concat(wakeup ? [AssistantLifecycleParticipant.WakeupGate] : [])
            .Concat(bargeIn ? [AssistantLifecycleParticipant.BargeIn] : [])
            .Concat([AssistantLifecycleParticipant.AudioFrontend]),
            composition.Lifecycle.StartOrder);
    }

    [Fact]
    public void ReadmeAndDemoExposeAllSupportedCompositionProfiles()
    {
        string repository = FindRepositoryRoot();
        string readme = File.ReadAllText(Path.Combine(repository, "README.md"));
        Assert.Contains("VoiceBox.Assistant.Hosting", readme, StringComparison.Ordinal);
        Assert.Contains(".UseAudio(audio)", readme, StringComparison.Ordinal);
        Assert.Contains("分别提供录音和播放服务", readme, StringComparison.Ordinal);
        Assert.Contains("AddWakeup", readme, StringComparison.Ordinal);
        Assert.Contains("AddSpeakerVerification", readme, StringComparison.Ordinal);
        Assert.Contains("AddBargeIn", readme, StringComparison.Ordinal);
        Assert.Contains("docs/voicebox-assistant-architecture.md", readme, StringComparison.Ordinal);

        // The README must not teach a constructor entry point that is no longer public.
        Assert.DoesNotContain("new VoiceAssistant(", readme, StringComparison.Ordinal);

        // Internal profile identifiers live in the linked architecture document. The public
        // README and current demo teach the Builder entry point instead of retired helpers.
        string architecture = File.ReadAllText(Path.Combine(repository, "docs", "voicebox-assistant-architecture.md"));
        Assert.Contains("BasicDuplexQa", architecture, StringComparison.Ordinal);
        Assert.Contains("BasicSplitAudioQa", architecture, StringComparison.Ordinal);
        string demoProfiles = ReadSource(
            "src",
            "Demos",
            "VoiceAssistantDemo",
            "Program.cs");
        foreach (string entryPoint in new[]
        {
            "VoiceAssistant.CreateBuilder()", ".UseAudio(", ".UseTurnBoundary(",
            ".UseSpeechRecognition(", ".UseResponseGeneration(", ".UseSpeechSynthesis(",
            ".AddWakeup(", ".AddSpeakerVerification(", ".AddBargeIn(",
        })
        {
            Assert.Contains(entryPoint, demoProfiles, StringComparison.Ordinal);
        }
    }

    private static IEnumerable<string> EnumerateSourceFiles(string root)
        => Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories)
            .Where(path => !path.Contains("\\bin\\", StringComparison.OrdinalIgnoreCase))
            .Where(path => !path.Contains("\\obj\\", StringComparison.OrdinalIgnoreCase));

    private static string ReadSource(params string[] segments)
    {
        string path = Path.Combine([FindRepositoryRoot(), .. segments]);
        return File.ReadAllText(path);
    }

    internal static string ReadRuntimeSource(params string[] relativePath)
        => File.ReadAllText(Path.Combine(
            [FindRepositoryRoot(), "src", "VoiceBox.Assistant", "Runtime", .. relativePath]));

    private static string FindRepositoryRoot()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "src", "VoiceBox.slnx")))
                return directory.FullName;
            directory = directory.Parent;
        }

        throw new Xunit.Sdk.XunitException("Repository root was not found.");
    }
}
