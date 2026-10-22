using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using System.Diagnostics;
using System.Reflection;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Composition.Exceptions;
using VoiceBox.Assistant.Composition.Graph;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Hosting;
using VoiceBox.Assistant.Hosting.Diagnostics;
using VoiceBox.Assistant.Hosting.Models;
using VoiceBox.Assistant.Runtime;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed partial class VoiceAssistantLifecycleTests
{
    [Fact]
    public void BuilderUsesModulePresence_NotOptionsEnableFlags()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantOptions
        {
            Wakeup = { Enable = true },
            BargeIn = { Enable = true },
        };

        AssistantComposition composition = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildComposition();

        Assert.DoesNotContain("WakeupGate", composition.Snapshot.Capabilities);
        Assert.DoesNotContain("SpeakerEvidence", composition.Snapshot.Capabilities);
        Assert.DoesNotContain("BargeIn", composition.Snapshot.Capabilities);
        Assert.Equal(
            [AssistantLifecycleParticipant.AudioFrontend],
            composition.Lifecycle.StartOrder);
    }

    [Fact]
    public void HostingNamespaceExposesDiHostedLifecycleAndSeparatedDiagnostics()
    {
        Assembly hosting = typeof(VoiceAssistantServiceCollectionExtensions).Assembly;
        Assert.Equal(typeof(VoiceAssistant).Assembly, hosting);

        Type extensions = RequireType(
            hosting,
            "VoiceBox.Assistant.Hosting.VoiceAssistantServiceCollectionExtensions");
        string[] methodNames = extensions.GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Select(method => method.Name)
            .ToArray();
        Assert.Contains("AddVoiceBoxAssistant", methodNames);
        Assert.Contains("AddVoiceBoxAssistantHostedLifecycle", methodNames);

        Type diagnostics = RequireType(
            hosting,
            "VoiceBox.Assistant.Hosting.VoiceAssistantDiagnostics");
        Assert.NotNull(diagnostics.GetProperty("RuntimeLifecycle"));
        Assert.NotNull(diagnostics.GetProperty("Turn"));
        Assert.NotNull(diagnostics.GetProperty("Modules"));
        Assert.Null(hosting.GetType("VoiceBox.Assistant.Hosting.AssistantContext"));
    }

    [Fact]
    public void BuilderAndDiCompositionHaveTheSameCapabilityGraphHash()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantOptions();
        using ServiceProvider provider = CreateProvider(services, options, hosted: false);

        VoiceAssistantDiagnostics diagnostics = provider.GetRequiredService<VoiceAssistantDiagnostics>();
        var direct = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .BuildWithDiagnostics();
        using (direct.Assistant)
        Assert.Equal(direct.CapabilityGraphHash, diagnostics.RuntimeLifecycle.CapabilityGraphHash);
        Assert.Equal("BasicDuplexQa", diagnostics.RuntimeLifecycle.ProfileId);
        Assert.Equal(64, diagnostics.RuntimeLifecycle.CapabilityGraphHash.Length);
    }

    /// <summary>
    /// SpeakerEvidence appears only when Confirmation.Enable is on and a speaker provider is
    /// registered. KWS without confirmation does not install an online identity module.
    /// </summary>
    [Theory]
    [InlineData(false, false, false, new string[] { "WakeupGate", "SpeakerEvidence", "BargeIn" })]
    [InlineData(false, true, false, new string[] { "WakeupGate", "SpeakerEvidence", "BargeIn" })]
    [InlineData(false, true, true, new string[] { "WakeupGate", "SpeakerEvidence" })]
    [InlineData(true, true, false, new string[] { "BargeIn", "SpeakerEvidence" })]
    [InlineData(true, true, true, new string[] { "SpeakerEvidence" })]
    public void DiOptionalModulesFollowTypedRegistrationPresence(
        bool wakeupRegistered,
        bool speakerRegistered,
        bool bargeInRegistered,
        string[] absentCapabilities)
    {
        var services = TestServices.Create();
        using ServiceProvider provider = CreateProvider(
            services,
            new VoiceAssistantOptions(),
            hosted: false,
            wakeupRegistered,
            speakerRegistered,
            bargeInRegistered);

        AssistantComposition composition = CreateHostingBuilder(provider).BuildComposition();
        var capabilities = new HashSet<string>(composition.Snapshot.Capabilities, StringComparer.Ordinal);

        foreach (string absent in absentCapabilities)
            Assert.DoesNotContain(absent, capabilities);
        if (wakeupRegistered)
            Assert.Contains("WakeupGate", capabilities);
        if (bargeInRegistered)
            Assert.Contains("BargeIn", capabilities);
    }

    [Fact]
    public void DiConfirmationEnabledWithoutSpeakerRegistration_FailsComposition()
    {
        var services = TestServices.Create();
        using ServiceProvider provider = CreateProvider(
            services,
            new VoiceAssistantOptions { Speaker = { Confirmation = { Enable = true } } },
            hosted: false,
            wakeupRegistered: true,
            speakerRegistered: false,
            bargeInRegistered: false);

        AssistantCompositionException failure = Assert.Throws<AssistantCompositionException>(
            () => CreateHostingBuilder(provider).BuildComposition());
        Assert.Equal("VBX-COMP-004", failure.Code);
    }

    [Fact]
    public void FullStackBuilderAndDiCompositionHaveTheSameCapabilityGraphHash()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantOptions
        {
            Wakeup = { Enable = true },
            Speaker = { Confirmation = { Enable = true } },
            BargeIn = { Enable = true },
        };
        using ServiceProvider provider = CreateProvider(
            services,
            options,
            hosted: false,
            wakeupRegistered: true,
            speakerRegistered: true,
            bargeInRegistered: true);

        AssistantComposition hosting = CreateHostingBuilder(provider).BuildComposition();
        AssistantComposition direct = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .AddWakeup(services.Wakeup)
            .AddSpeakerVerification(services.Speaker)
            .AddBargeIn(
                options.BargeIn,
                TestAcousticPathDelayEstimatorFactory.Instance)
            .BuildComposition();
        Assert.Equal(direct.Snapshot.CapabilityGraphHash, hosting.Snapshot.CapabilityGraphHash);
    }

    [Fact]
    public void CapabilityGraphHashUsesSortedCapabilityIdsAndSha256()
    {
        string hash = AssistantCapabilityGraph.ComputeCapabilityGraphHash(
            ["SpeechSynthesis", "AudioCapture"]);

        Assert.Equal(
            "6a20c9278b9a18a3b2347c57de0ec5736746c14ebe2c95a847852c31dd777159",
            hash);
    }

    [Fact]
    public void DiOptionsValidationUsesTheExistingOracle()
    {
        var registrations = new ServiceCollection();
        registrations.AddVoiceBoxAssistant(options => options.MinSentenceChars = 0);
        using ServiceProvider provider = registrations.BuildServiceProvider();

        OptionsValidationException exception = Assert.Throws<OptionsValidationException>(
            () => provider.GetRequiredService<IOptions<VoiceAssistantOptions>>().Value);
        Assert.Contains("MinSentenceChars must be >= 1.", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DiMissingRequiredCapabilityFailsBeforeAnyProviderStarts()
    {
        var services = TestServices.Create();
        var registrations = new ServiceCollection();
        registrations.AddLogging();
        registrations.AddSingleton<IAudioDuplexService>(services.Audio);
        registrations.AddSingleton<IVoiceActivityDetectService>(services.Vad);
        registrations.AddSingleton<IASRService>(services.Asr);
        registrations.AddSingleton<IStreamLLMService>(services.Llm);
        registrations.AddVoiceBoxAssistant();
        using ServiceProvider provider = registrations.BuildServiceProvider();

        InvalidOperationException exception = Assert.Throws<InvalidOperationException>(
            () => provider.GetRequiredService<VoiceAssistant>());

        Assert.Contains(nameof(IStreamTTSService), exception.Message, StringComparison.Ordinal);
        Assert.Equal(0, services.Audio.StartCount);
    }

    [Fact]
    public void DiUsesSplitAudioWhenNoDuplexProviderIsRegistered()
    {
        var services = TestServices.Create();
        var record = new FakeDuplexService();
        var playback = new FakeDuplexService();
        var registrations = new ServiceCollection();
        registrations.AddLogging();
        registrations.AddSingleton<IAudioRecordService>(record);
        registrations.AddSingleton<IAudioPlayService>(playback);
        registrations.AddSingleton<IVoiceActivityDetectService>(services.Vad);
        registrations.AddSingleton<IASRService>(services.Asr);
        registrations.AddSingleton<IStreamLLMService>(services.Llm);
        registrations.AddSingleton<IStreamTTSService>(services.Tts);
        registrations.AddVoiceBoxAssistant();
        using ServiceProvider provider = registrations.BuildServiceProvider();

        AssistantComposition composition = CreateHostingBuilder(provider).BuildComposition();

        Assert.Equal("BasicSplitAudioQa", composition.Snapshot.ProfileId);
        Assert.DoesNotContain("SimultaneousCapturePlayback", composition.Snapshot.Capabilities);
    }

    [Fact]
    public void RuntimeOwnsASettingsSnapshot_NotTheMutableCallerOptions()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantOptions { MinSentenceChars = 9 };
        using VoiceAssistant assistant = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseAudio(services.Audio)
            .UseTurnBoundary(services.Vad)
            .UseSpeechRecognition(services.Asr)
            .UseResponseGeneration(services.Llm)
            .UseSpeechSynthesis(services.Tts)
            .Build();

        options.MinSentenceChars = 99;
        AssistantRuntime runtime = (AssistantRuntime)typeof(VoiceAssistant)
            .GetField("_runtime", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(assistant)!;
        var snapshot = RuntimeGraphProbe.RequireField<VoiceAssistantOptions>(runtime, "_options");

        Assert.NotSame(options, snapshot);
        Assert.Equal(9, snapshot.MinSentenceChars);
    }

    [Fact]
    public async Task HostedTelemetryEnabledAndDisabledPreserveLifecycleTrace()
    {
        string[] withoutListener = await RunHostedLifecycleTrace(withActivityListener: false);
        string[] withListener = await RunHostedLifecycleTrace(withActivityListener: true);

        Assert.Equal(withoutListener, withListener);
        Assert.Equal("Stopped", withListener[^1]);
    }

    [Fact]
    public async Task HostedStartCancellationIsPropagatedWithoutChangingTheCancellationException()
    {
        var services = TestServices.Create();
        services.Audio.StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.StartBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using ServiceProvider provider = CreateProvider(services, new VoiceAssistantOptions(), hosted: true);
        IHostedService hosted = Assert.Single(provider.GetServices<IHostedService>());
        VoiceAssistantDiagnostics diagnostics = provider.GetRequiredService<VoiceAssistantDiagnostics>();
        using var cancellation = new CancellationTokenSource();

        Task start = hosted.StartAsync(cancellation.Token);
        await services.Audio.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(1));
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => start);
        Assert.Equal("Faulted", diagnostics.RuntimeLifecycle.Status);
    }

    [Fact]
    public async Task HostedStartFaultIsPropagatedWithoutChangingTheException()
    {
        var services = TestServices.Create();
        services.Audio.StartEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.StartBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using ServiceProvider provider = CreateProvider(services, new VoiceAssistantOptions(), hosted: true);
        IHostedService hosted = Assert.Single(provider.GetServices<IHostedService>());
        VoiceAssistantDiagnostics diagnostics = provider.GetRequiredService<VoiceAssistantDiagnostics>();

        Task start = hosted.StartAsync(CancellationToken.None);
        await services.Audio.StartEntered.Task.WaitAsync(TimeSpan.FromSeconds(1));
        services.Audio.StartBlock.SetException(new InvalidOperationException("hosted-start-failure"));

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() => start);
        Assert.Equal("hosted-start-failure", exception.Message);
        Assert.Equal("Faulted", diagnostics.RuntimeLifecycle.Status);
    }

    [Fact]
    public async Task HostedStopPreservesCoreCancellationSemantics()
    {
        var services = TestServices.Create();
        using ServiceProvider provider = CreateProvider(services, new VoiceAssistantOptions(), hosted: true);
        IHostedService hosted = Assert.Single(provider.GetServices<IHostedService>());
        VoiceAssistantDiagnostics diagnostics = provider.GetRequiredService<VoiceAssistantDiagnostics>();
        await hosted.StartAsync(CancellationToken.None);

        services.Audio.StopEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        services.Audio.StopBlock = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var cancellation = new CancellationTokenSource();
        Task stop = hosted.StopAsync(cancellation.Token);
        await services.Audio.StopEntered.Task.WaitAsync(TimeSpan.FromSeconds(1));
        cancellation.Cancel();
        await Task.Delay(25);
        Assert.False(stop.IsCompleted);
        Assert.Equal("Stopping", diagnostics.RuntimeLifecycle.Status);

        services.Audio.StopBlock.SetResult();
        await stop;
        Assert.Equal("Stopped", diagnostics.RuntimeLifecycle.Status);
    }

    [Fact]
    public void DiagnosticsContainOnlySanitizedIdentityAndStatus()
    {
        var services = TestServices.Create();
        var options = new VoiceAssistantOptions { SystemPrompt = "sensitive-prompt" };
        using ServiceProvider provider = CreateProvider(services, options, hosted: false);
        VoiceAssistantDiagnostics diagnostics = provider.GetRequiredService<VoiceAssistantDiagnostics>();

        string serialized = System.Text.Json.JsonSerializer.Serialize(new
        {
            diagnostics.RuntimeLifecycle,
            diagnostics.Turn,
            diagnostics.Modules,
        });
        Assert.DoesNotContain(options.SystemPrompt, serialized, StringComparison.Ordinal);
        Assert.DoesNotContain(services.Audio.GetType().FullName!, serialized, StringComparison.Ordinal);
        Assert.DoesNotContain("IServiceProvider", serialized, StringComparison.Ordinal);
    }

    [Fact]
    public void RuntimeAndDiagnosticsKeepGraphProviderAndModuleHealthSeparated()
    {
        FieldInfo[] runtimeFields = ShellSurface.Fields;
        Assert.DoesNotContain(runtimeFields, field =>
            field.FieldType == typeof(AssistantCapabilityGraph) ||
            field.FieldType == typeof(IServiceProvider) ||
            field.Name.Contains("graph", StringComparison.OrdinalIgnoreCase) ||
            field.Name.Contains("serviceProvider", StringComparison.OrdinalIgnoreCase));

        PropertyInfo[] diagnosticsProperties = typeof(VoiceAssistantDiagnostics).GetProperties(
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.DeclaredOnly);
        Assert.Equal(
            ["Modules", "RuntimeLifecycle", "Turn"],
            diagnosticsProperties.Select(property => property.Name).Order(StringComparer.Ordinal));
        Assert.Equal(typeof(VoiceAssistantRuntimeLifecycleSnapshot),
            typeof(VoiceAssistantDiagnostics).GetProperty("RuntimeLifecycle")!.PropertyType);
        Assert.Equal(typeof(VoiceAssistantTurnSnapshot),
            typeof(VoiceAssistantDiagnostics).GetProperty("Turn")!.PropertyType);
        Assert.Equal(typeof(VoiceAssistantModuleHealthSnapshot),
            typeof(VoiceAssistantDiagnostics).GetProperty("Modules")!.PropertyType);
    }

    private static async Task<string[]> RunHostedLifecycleTrace(bool withActivityListener)
    {
        ActivityListener? listener = null;
        if (withActivityListener)
        {
            listener = new ActivityListener
            {
                ShouldListenTo = source => source.Name == "VoiceBox.Assistant.Hosting",
                Sample = static (ref ActivityCreationOptions<ActivityContext> _) =>
                    ActivitySamplingResult.AllData,
            };
            ActivitySource.AddActivityListener(listener);
        }

        try
        {
            var services = TestServices.Create();
            using ServiceProvider provider = CreateProvider(
                services,
                new VoiceAssistantOptions(),
                hosted: true);
            VoiceAssistantDiagnostics diagnostics = provider.GetRequiredService<VoiceAssistantDiagnostics>();
            VoiceAssistant assistant = provider.GetRequiredService<VoiceAssistant>();
            IHostedService hosted = Assert.Single(provider.GetServices<IHostedService>());
            var trace = new List<string> { diagnostics.RuntimeLifecycle.Status };
            assistant.StateChanged += (_, args) => trace.Add($"state:{args.To}");
            await hosted.StartAsync(CancellationToken.None);
            trace.Add(diagnostics.RuntimeLifecycle.Status);
            await hosted.StopAsync(CancellationToken.None);
            trace.Add(diagnostics.RuntimeLifecycle.Status);
            return trace.ToArray();
        }
        finally
        {
            listener?.Dispose();
        }
    }

    private static ServiceProvider CreateProvider(
        TestServices services,
        VoiceAssistantOptions options,
        bool hosted,
        bool wakeupRegistered = false,
        bool speakerRegistered = false,
        bool bargeInRegistered = false)
    {
        var registrations = new ServiceCollection();
        registrations.AddLogging();
        registrations.AddSingleton<IAudioDuplexService>(services.Audio);
        registrations.AddSingleton<IAudioRecordService>(services.Audio);
        registrations.AddSingleton<IAudioPlayService>(services.Audio);
        registrations.AddSingleton<IVoiceActivityDetectService>(services.Vad);
        registrations.AddSingleton<IASRService>(services.Asr);
        registrations.AddSingleton<IStreamLLMService>(services.Llm);
        registrations.AddSingleton<IStreamTTSService>(services.Tts);
        if (wakeupRegistered)
            registrations.AddSingleton<IWakeupService>(services.Wakeup);
        if (speakerRegistered)
            registrations.AddSingleton<ISpeakerService>(services.Speaker);
        registrations.AddVoiceBoxAssistant(target =>
        {
            target.MinSentenceChars = options.MinSentenceChars;
            target.SystemPrompt = options.SystemPrompt;
            target.Speaker.ProfilePath = options.Speaker.ProfilePath;
            target.Speaker.Confirmation.Enable = options.Speaker.Confirmation.Enable;
        });
        if (bargeInRegistered)
        {
            registrations.AddSingleton<IAcousticPathDelayEstimatorFactory>(
                TestAcousticPathDelayEstimatorFactory.Instance);
            registrations.AddVoiceBoxAssistantBargeIn();
        }
        if (hosted)
            registrations.AddVoiceBoxAssistantHostedLifecycle();
        return registrations.BuildServiceProvider();
    }

    private static VoiceAssistantBuilder CreateHostingBuilder(IServiceProvider provider)
    {
        MethodInfo method = typeof(VoiceAssistantServiceCollectionExtensions).GetMethod(
            "CreateBuilder",
            BindingFlags.Static | BindingFlags.NonPublic)
            ?? throw new Xunit.Sdk.XunitException("Hosting composition factory was not found.");
        return (VoiceAssistantBuilder)(method.Invoke(null, [provider])
            ?? throw new Xunit.Sdk.XunitException("Hosting composition factory returned null."));
    }

    private static Type RequireType(Assembly assembly, string fullName)
        => assembly.GetType(fullName) ?? throw new Xunit.Sdk.XunitException(
            $"Required type '{fullName}' was not found.");
}
