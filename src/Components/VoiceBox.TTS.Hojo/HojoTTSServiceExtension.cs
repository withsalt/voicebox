using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.TTSService;

namespace VoiceBox.TTS.Hojo;

public static class HojoTTSServiceExtension
{
    public static AudioServiceManager AddHojoTTSService(this AudioServiceManager services, Action<HojoTTSOptions> configure, string? serviceName = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        if (string.IsNullOrWhiteSpace(serviceName))
        {
            services.AddService<IStreamTTSService, HojoTTSService>().Configure(configure);
            services.AddService<ITTSService>(sc => sc.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>()));
        }
        else
        {
            services.AddService<IStreamTTSService, HojoTTSService>(serviceName).Configure(configure);
            services.AddService<ITTSService>(sc => sc.AddKeyedSingleton<ITTSService>(serviceName, (sp, _) => sp.GetRequiredKeyedService<IStreamTTSService>(serviceName)));
        }
        return services;
    }

    public static IServiceCollection AddHojoTTSService(this IServiceCollection services, Action<HojoTTSOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        services.Configure(configure);
        services.TryAddSingleton<HojoTTSService>();
        services.TryAddSingleton<IStreamTTSService>(sp => sp.GetRequiredService<HojoTTSService>());
        services.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<HojoTTSService>());
        return services;
    }
}
