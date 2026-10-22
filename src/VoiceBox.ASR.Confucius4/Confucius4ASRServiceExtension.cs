using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.ASRService;

namespace VoiceBox.ASR.Confucius4;

public static class Confucius4ASRServiceExtension
{
    public static IServiceCollection AddConfucius4ASRService(this IServiceCollection services,
        Action<Confucius4AsrOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        services.Configure(configure);
        services.TryAddSingleton<Confucius4ASRService>();
        services.TryAddSingleton<IStreamASRService>(sp => sp.GetRequiredService<Confucius4ASRService>());
        services.TryAddSingleton<IASRService>(sp => sp.GetRequiredService<Confucius4ASRService>());
        return services;
    }

    public static AudioServiceManager AddConfucius4ASRService(this AudioServiceManager services,
        Action<Confucius4AsrOptions> configure, string? serviceName = "")
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);
        if (string.IsNullOrEmpty(serviceName))
        {
            services.AddService<IStreamASRService, Confucius4ASRService>().Configure(configure);
            services.AddService<IASRService>(s => s.TryAddSingleton<IASRService>(sp => sp.GetRequiredService<IStreamASRService>()));
        }
        else
        {
            services.AddService<IStreamASRService, Confucius4ASRService>(serviceName).Configure(configure);
            services.AddService<IASRService>(s => s.AddKeyedSingleton<IASRService>(serviceName,
                (sp, _) => sp.GetRequiredKeyedService<IStreamASRService>(serviceName)));
        }
        return services;
    }
}
