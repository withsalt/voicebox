using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Hosting.Configuration;
using VoiceBox.Assistant.Hosting.Diagnostics;
using VoiceBox.Assistant.Hosting.Lifecycle;
using VoiceBox.Assistant.Hosting.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;

namespace VoiceBox.Assistant.Hosting;

public static class VoiceAssistantServiceCollectionExtensions
{
    public static IServiceCollection AddVoiceBoxAssistant(
        this IServiceCollection services,
        Action<VoiceAssistantOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(services);

        OptionsBuilder<VoiceAssistantOptions> options = services.AddOptions<VoiceAssistantOptions>();
        if (configure is not null)
            options.Configure(configure);
        options.ValidateOnStart();

        services.TryAddEnumerable(
            ServiceDescriptor.Singleton<
                IValidateOptions<VoiceAssistantOptions>,
                VoiceAssistantOptionsValidateOnStart>());
        services.TryAddSingleton<VoiceAssistantHostInstance>(CreateHostInstance);
        services.TryAddSingleton(static provider =>
            provider.GetRequiredService<VoiceAssistantHostInstance>().Assistant);
        services.TryAddSingleton(static provider =>
            provider.GetRequiredService<VoiceAssistantHostInstance>().Diagnostics);
        return services;
    }

    public static IServiceCollection AddVoiceBoxAssistantBargeIn(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.TryAddSingleton<VoiceAssistantBargeInModuleRegistration>();
        return services;
    }

    public static IServiceCollection AddVoiceBoxAssistantHostedLifecycle(
        this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.TryAddEnumerable(
            ServiceDescriptor.Singleton<IHostedService, VoiceAssistantHostedLifecycle>());
        return services;
    }

    private static VoiceAssistantHostInstance CreateHostInstance(IServiceProvider provider)
    {
        VoiceAssistantBuilder builder = CreateBuilder(provider);
        var result = builder.BuildWithDiagnostics(provider.GetService<ILogger<VoiceAssistant>>());
        var diagnostics = new VoiceAssistantDiagnostics(
            result.Assistant,
            result.ProfileId,
            result.CapabilityGraphHash,
            result.Capabilities);
        return new VoiceAssistantHostInstance(result.Assistant, diagnostics);
    }

    private static VoiceAssistantBuilder CreateBuilder(IServiceProvider provider)
    {
        VoiceAssistantOptions options = provider
            .GetRequiredService<IOptions<VoiceAssistantOptions>>()
            .Value;
        VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
            .UseOptions(options)
            .UseTurnBoundary(provider.GetRequiredService<IVoiceActivityDetectService>())
            .UseSpeechRecognition(provider.GetRequiredService<IASRService>())
            .UseResponseGeneration(provider.GetRequiredService<IStreamLLMService>())
            .UseSpeechSynthesis(provider.GetRequiredService<IStreamTTSService>());

        IAudioDuplexService? duplex = provider.GetService<IAudioDuplexService>();
        if (duplex is not null)
        {
            builder.UseAudio(duplex);
        }
        else
        {
            builder.UseAudio(
                provider.GetRequiredService<IAudioRecordService>(),
                provider.GetRequiredService<IAudioPlayService>());
        }

        IWakeupService? wakeup = provider.GetService<IWakeupService>();
        if (wakeup is not null)
            builder.AddWakeup(wakeup);

        ISpeakerService? speaker = provider.GetService<ISpeakerService>();
        if (speaker is not null)
            builder.AddSpeakerVerification(speaker);

        IAcousticProcessingMetricsSource? acousticDiagnostics =
            provider.GetService<IAcousticProcessingMetricsSource>();
        if (acousticDiagnostics is not null)
            builder.AddAcousticDiagnostics(acousticDiagnostics);

        if (provider.GetService<VoiceAssistantBargeInModuleRegistration>() is not null)
        {
            builder.AddBargeIn(
                options.BargeIn,
                provider.GetRequiredService<IAcousticPathDelayEstimatorFactory>());
        }

        return builder;
    }
}
