using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Core;
using VoiceBox.Core.Services.VadService;
using VoiceBox.VAD.Silero.Models;

namespace VoiceBox.VAD.Silero
{
    public static class SileroVoiceActivityDetectServiceExtension
    {
        public static AudioServiceManager AddSileroVoiceActivityDetectService(this AudioServiceManager services, Action<SileroVadOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IVoiceActivityDetectService, SileroVoiceActivityDetectService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IVoiceActivityDetectService, SileroVoiceActivityDetectService>().Configure(configure);
            }
            return services;
        }

        public static IServiceCollection AddSileroVoiceActivityDetectService(this IServiceCollection services, Action<SileroVadOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IVoiceActivityDetectService, SileroVoiceActivityDetectService>();
            return services;
        }
    }
}
