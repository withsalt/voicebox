using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Core;
using VoiceBox.Core.Services.TTSService;

namespace VoiceBox.TTS.Kokoro
{
    public static class KokoroTTSServiceExtension
    {
        public static AudioServiceManager AddKokoroTTSService(this AudioServiceManager services, Action<KokoroTTSOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IStreamTTSService, KokoroTTSService>(serviceName).Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.AddKeyedSingleton<ITTSService>(serviceName, (sp, _) =>
                        sp.GetRequiredKeyedService<IStreamTTSService>(serviceName)));
            }
            else
            {
                services.AddService<IStreamTTSService, KokoroTTSService>().Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>()));
            }
            return services;
        }

        public static IServiceCollection AddKokoroTTSService(this IServiceCollection services, Action<KokoroTTSOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IStreamTTSService, KokoroTTSService>();
            services.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>());
            return services;
        }
    }
}
