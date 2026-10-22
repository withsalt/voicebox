using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Core;
using VoiceBox.Core.Services.TTSService;

namespace VoiceBox.TTS.MossNano
{
    public static class MossNanoTTSServiceExtension
    {
        public static AudioServiceManager AddMossNanoTTSService(this AudioServiceManager services, Action<MossNanoTTSOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IStreamTTSService, MossNanoTTSService>(serviceName).Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.AddKeyedSingleton<ITTSService>(serviceName, (sp, _) =>
                        sp.GetRequiredKeyedService<IStreamTTSService>(serviceName)));
            }
            else
            {
                services.AddService<IStreamTTSService, MossNanoTTSService>().Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>()));
            }
            return services;
        }

        public static IServiceCollection AddMossNanoTTSService(this IServiceCollection services, Action<MossNanoTTSOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IStreamTTSService, MossNanoTTSService>();
            services.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>());
            return services;
        }
    }
}
