using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.DTD.RenderAware.Models;

namespace VoiceBox.DTD.RenderAware;

/// <summary>Dependency-injection registration for the render-aware DTD service.</summary>
public static class RenderAwareDtdServiceExtension
{
    /// <summary>
    /// Registers one singleton detector. A singleton is appropriate for one audio stream; register
    /// separate keyed instances when an application processes multiple streams.
    /// </summary>
    public static IServiceCollection AddRenderAwareDtdService(
        this IServiceCollection services,
        Action<RenderAwareDtdOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        services.Configure(configure);
        services.TryAddSingleton<IRenderAwareDtdService, RenderAwareDtdService>();
        return services;
    }
}
