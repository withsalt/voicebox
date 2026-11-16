using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.HPFService;

namespace VoiceBox.HPF.Butterworth;

/// <summary>
/// Provides dependency injection registrations for the Butterworth high-pass filter.
/// </summary>
public static class ButterworthHighPassFilterServiceExtensions
{
    /// <summary>
    /// Registers a Butterworth high-pass filter with an <see cref="AudioServiceManager"/>.
    /// </summary>
    public static AudioServiceManager AddButterworthHighPassFilterService(
        this AudioServiceManager services,
        Action<ButterworthHighPassFilterOptions> configure,
        string? serviceName = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        string? normalizedServiceName = string.IsNullOrWhiteSpace(serviceName) ? null : serviceName;
        services
            .AddService<IHighPassFilterService, ButterworthHighPassFilterService>(normalizedServiceName)
            .Configure(configure);
        return services;
    }

    /// <summary>
    /// Registers a singleton Butterworth high-pass filter with an <see cref="IServiceCollection"/>.
    /// </summary>
    public static IServiceCollection AddButterworthHighPassFilterService(
        this IServiceCollection services,
        Action<ButterworthHighPassFilterOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configure);

        services.Configure(configure);
        services.TryAddSingleton<IHighPassFilterService, ButterworthHighPassFilterService>();
        return services;
    }
}
