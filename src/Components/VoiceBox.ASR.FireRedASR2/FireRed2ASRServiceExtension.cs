using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Core;
using VoiceBox.Core.Services.ASRService;

namespace VoiceBox.ASR.FireRedASR2
{
    public static class FireRed2ASRServiceExtension
    {
        public static AudioServiceManager AddFireRed2ASRService(this AudioServiceManager services, Action<FireRed2ASROptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IASRService, FireRed2ASRService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IASRService, FireRed2ASRService>().Configure(configure);
            }
            return services;
        }

        public static IServiceCollection AddFireRed2ASRService(this IServiceCollection services, Action<FireRed2ASROptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IASRService, FireRed2ASRService>();
            return services;
        }
    }
}
