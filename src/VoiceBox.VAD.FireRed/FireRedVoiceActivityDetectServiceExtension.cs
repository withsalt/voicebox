using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Core;
using VoiceBox.Core.Services.VadService;
using VoiceBox.VAD.FireRed.Models;

namespace VoiceBox.VAD.FireRed
{
    public static class FireRedVoiceActivityDetectServiceExtension
    {
        public static AudioServiceManager AddFireRedVoiceActivityDetectService(this AudioServiceManager services, Action<FireRedVadOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IVoiceActivityDetectService, FireRedVoiceActivityDetectService>(serviceName).Configure(configure);

            }
            else
            {
                services.AddService<IVoiceActivityDetectService, FireRedVoiceActivityDetectService>().Configure(configure);

            }
            return services;
        }

        public static IServiceCollection AddFireRedVoiceActivityDetectService(this IServiceCollection services, Action<FireRedVadOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IVoiceActivityDetectService, FireRedVoiceActivityDetectService>();
            return services;
        }
    }
}
