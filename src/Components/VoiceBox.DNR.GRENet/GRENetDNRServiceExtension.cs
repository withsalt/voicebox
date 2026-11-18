using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.DNR.GRENet.Models;

namespace VoiceBox.DNR.GRENet;

/// <summary>Dependency injection extensions for GRE-Net DNR.</summary>
public static class GRENetDNRServiceExtension
{
    /// <summary>Registers a GRE-Net service with the audio service manager.</summary>
    public static AudioServiceManager AddGRENetDNRService(
        this AudioServiceManager services,
        Action<GRENetOptions> configure,
        string? serviceName = "")
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        if (!string.IsNullOrWhiteSpace(serviceName))
        {
            services.AddService<IDigitalNoiseReductionService, GRENetDNRService>(serviceName).Configure(configure);
        }
        else
        {
            services.AddService<IDigitalNoiseReductionService, GRENetDNRService>().Configure(configure);
        }

        return services;
    }

    /// <summary>Registers GRE-Net as the default singleton DNR service.</summary>
    public static IServiceCollection AddGRENetDNRService(this IServiceCollection services, Action<GRENetOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        services.Configure(configure);
        services.TryAddSingleton<IDigitalNoiseReductionService, GRENetDNRService>();
        return services;
    }
}
