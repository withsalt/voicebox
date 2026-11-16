using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Core;
using VoiceBox.Core.Services.DNRService;

namespace VoiceBox.DNR.RNNoise
{
    public static class RNNoiseDNRServiceExtension
    {
        public static AudioServiceManager AddRNNoiseDNRService(this AudioServiceManager services, Action<RNNoiseOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IDigitalNoiseReductionService, RNNoiseDNRService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IDigitalNoiseReductionService, RNNoiseDNRService>().Configure(configure);
            }
            return services;
        }

        public static IServiceCollection AddRNNoiseDNRService(this IServiceCollection services, Action<RNNoiseOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IDigitalNoiseReductionService, RNNoiseDNRService>();
            return services;
        }
    }
}
