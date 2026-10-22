using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.AEC.DTLN.Models;
using VoiceBox.Core;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;

namespace VoiceBox.AEC.DTLN
{
    public static class DtlnAecServiceExtension
    {
        public static AudioServiceManager AddDtlnAecService(
            this AudioServiceManager services,
            Action<DtlnAecOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
                services.AddService<IAcousticEchoCancellationService, DtlnAecService>(serviceName).Configure(configure);
            else
                services.AddService<IAcousticEchoCancellationService, DtlnAecService>().Configure(configure);

            return services;
        }

        public static IServiceCollection AddDtlnAecService(
            this IServiceCollection services,
            Action<DtlnAecOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            bool processorAlreadyRegistered = services.Any(descriptor =>
                descriptor.ServiceType == typeof(IAcousticEchoCancellationService) &&
                descriptor.ServiceKey is null);
            if (!processorAlreadyRegistered)
            {
                services.AddSingleton<IAcousticEchoCancellationService, DtlnAecService>();
                services.TryAddSingleton<IAcousticProcessingMetricsSource>(provider =>
                    (IAcousticProcessingMetricsSource)provider.GetRequiredService<IAcousticEchoCancellationService>());
            }
            return services;
        }
    }
}
