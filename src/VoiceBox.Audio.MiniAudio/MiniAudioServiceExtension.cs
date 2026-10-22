using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.MiniAudio.Models;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService;

namespace VoiceBox.Audio.MiniAudio;

public static class MiniAudioServiceExtension
{
    public static AudioServiceManager AddMiniAudioRecordService(
        this AudioServiceManager services,
        Action<MiniAudioRecordOptions> configure,
        string? serviceName = "")
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        if (string.IsNullOrWhiteSpace(serviceName))
        {
            services.AddService<IAudioRecordService, MiniAudioRecordService>().Configure(configure);
        }
        else
        {
            services.AddService<IAudioRecordService, MiniAudioRecordService>(serviceName).Configure(configure);
        }

        return services;
    }

    public static AudioServiceManager AddMiniAudioPlayService(
        this AudioServiceManager services,
        Action<MiniAudioPlayOptions> configure,
        string? serviceName = "")
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        if (string.IsNullOrWhiteSpace(serviceName))
        {
            services.AddService<IAudioPlayService, MiniAudioPlayService>().Configure(configure);
        }
        else
        {
            services.AddService<IAudioPlayService, MiniAudioPlayService>(serviceName).Configure(configure);
        }

        return services;
    }

    public static IServiceCollection AddMiniAudioRecordService(
        this IServiceCollection services,
        Action<MiniAudioRecordOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        services.Configure(configure);
        services.TryAddSingleton<IAudioRecordService, MiniAudioRecordService>();
        return services;
    }

    public static IServiceCollection AddMiniAudioPlayService(
        this IServiceCollection services,
        Action<MiniAudioPlayOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        services.Configure(configure);
        services.TryAddSingleton<IAudioPlayService, MiniAudioPlayService>();
        return services;
    }

    public static AudioServiceManager AddMiniAudioDuplexService(
        this AudioServiceManager services,
        Action<MiniAudioDuplexOptions> configure,
        string? serviceName = "")
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        return services.AddService<IAudioRecordService>(serviceCollection =>
        {
            if (string.IsNullOrWhiteSpace(serviceName))
            {
                serviceCollection.Configure(configure);
                serviceCollection.TryAddSingleton<IAudioDuplexService, MiniAudioDuplexService>();
                serviceCollection.TryAddSingleton<IAudioRecordService>(provider => provider.GetRequiredService<IAudioDuplexService>());
                serviceCollection.TryAddSingleton<IAudioPlayService>(provider => provider.GetRequiredService<IAudioDuplexService>());
                return;
            }

            serviceCollection.Configure(serviceName, configure);
            serviceCollection.AddKeyedSingleton<IAudioDuplexService>(serviceName, (provider, _) =>
            {
                MiniAudioDuplexOptions namedOptions = provider
                    .GetRequiredService<IOptionsMonitor<MiniAudioDuplexOptions>>()
                    .Get(serviceName);
                return ActivatorUtilities.CreateInstance<MiniAudioDuplexService>(
                    provider,
                    new OptionsWrapper<MiniAudioDuplexOptions>(namedOptions));
            });
            serviceCollection.AddKeyedSingleton<IAudioRecordService>(serviceName, (provider, _) =>
                provider.GetRequiredKeyedService<IAudioDuplexService>(serviceName));
            serviceCollection.AddKeyedSingleton<IAudioPlayService>(serviceName, (provider, _) =>
                provider.GetRequiredKeyedService<IAudioDuplexService>(serviceName));
        });
    }

    public static IServiceCollection AddMiniAudioFullDuplexService(
        this IServiceCollection services,
        Action<MiniAudioDuplexOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        services.Configure(configure);
        services.TryAddSingleton<IAudioDuplexService, MiniAudioDuplexService>();
        services.TryAddSingleton<IAudioRecordService>(provider => provider.GetRequiredService<IAudioDuplexService>());
        services.TryAddSingleton<IAudioPlayService>(provider => provider.GetRequiredService<IAudioDuplexService>());
        return services;
    }
}
