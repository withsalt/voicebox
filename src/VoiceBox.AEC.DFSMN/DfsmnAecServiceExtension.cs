using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using VoiceBox.AEC.DFSMN.Models;
using VoiceBox.Core;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.VadService;

namespace VoiceBox.AEC.DFSMN
{
    public static class DfsmnAecServiceExtension
    {
        public static AudioServiceManager AddDfsmnAecService(
            this AudioServiceManager services,
            Action<DfsmnAecOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IAcousticEchoCancellationService, DfsmnAecService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IAcousticEchoCancellationService, DfsmnAecService>().Configure(configure);
            }

            return services;
        }

        public static IServiceCollection AddDfsmnAecService(
            this IServiceCollection services,
            Action<DfsmnAecOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            bool processorAlreadyRegistered = services.Any(descriptor =>
                descriptor.ServiceType == typeof(IAcousticEchoCancellationService) &&
                descriptor.ServiceKey is null);
            if (!processorAlreadyRegistered)
            {
                services.AddSingleton<IAcousticEchoCancellationService, DfsmnAecService>();
                services.TryAddSingleton<IAcousticProcessingMetricsSource>(provider =>
                    (IAcousticProcessingMetricsSource)provider.GetRequiredService<IAcousticEchoCancellationService>());
            }
            return services;
        }

        public static AudioServiceManager AddDfsmnAecVadService(
            this AudioServiceManager services,
            Action<DfsmnAecVadOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.AddService<IAcousticEchoCancellationService>(collection =>
                AddDfsmnAecVadRegistrations(collection, configure, serviceName));
            return services;
        }

        public static IServiceCollection AddDfsmnAecVadService(
            this IServiceCollection services,
            Action<DfsmnAecVadOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            AddDfsmnAecVadRegistrations(services, configure, serviceName: "");
            return services;
        }

        private static void AddDfsmnAecVadRegistrations(
            IServiceCollection services,
            Action<DfsmnAecVadOptions> configure,
            string? serviceName)
        {
            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.Configure(serviceName, configure);
                services.AddKeyedSingleton<DfsmnAecVadService>(serviceName, (sp, key) =>
                {
                    var monitor = sp.GetRequiredService<IOptionsMonitor<DfsmnAecVadOptions>>();
                    var options = new OptionsWrapper<DfsmnAecVadOptions>(monitor.Get((string)key!));
                    return ActivatorUtilities.CreateInstance<DfsmnAecVadService>(sp, options);
                });
                services.AddKeyedSingleton<IAcousticEchoCancellationService>(
                    serviceName,
                    (sp, key) => sp.GetRequiredKeyedService<DfsmnAecVadService>(key));
                services.AddKeyedSingleton<IAcousticProcessingMetricsSource>(
                    serviceName,
                    (sp, key) => sp.GetRequiredKeyedService<DfsmnAecVadService>(key));
                services.AddKeyedSingleton<IVoiceActivityDetectService>(
                    serviceName,
                    (sp, key) => sp.GetRequiredKeyedService<DfsmnAecVadService>(key));
                return;
            }

            services.Configure(configure);
            services.TryAddSingleton<DfsmnAecVadService>();
            services.AddSingleton<IAcousticEchoCancellationService>(sp => sp.GetRequiredService<DfsmnAecVadService>());
            services.AddSingleton<IAcousticProcessingMetricsSource>(sp => sp.GetRequiredService<DfsmnAecVadService>());
            services.AddSingleton<IVoiceActivityDetectService>(sp => sp.GetRequiredService<DfsmnAecVadService>());
        }

    }
}
