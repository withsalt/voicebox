using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.ASRService;

namespace VoiceBox.ASR.Qwen3
{
    public static class Qwen3ASRServiceExtension
    {
        public static AudioServiceManager AddQwen3ASRService(
            this AudioServiceManager services,
            Action<Qwen3ASROptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IStreamASRService, Qwen3ASRService>(serviceName).Configure(configure);
                services.AddService<IASRService>(sc =>
                    sc.AddKeyedSingleton<IASRService>(serviceName, (sp, _) =>
                        sp.GetRequiredKeyedService<IStreamASRService>(serviceName)));
            }
            else
            {
                services.AddService<IStreamASRService, Qwen3ASRService>().Configure(configure);
                services.AddService<IASRService>(sc =>
                    sc.TryAddSingleton<IASRService>(sp => sp.GetRequiredService<IStreamASRService>()));
            }
            return services;
        }

        public static IServiceCollection AddQwen3ASRService(
            this IServiceCollection services,
            Action<Qwen3ASROptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            services.Configure(configure);
            services.TryAddSingleton<IStreamASRService, Qwen3ASRService>();
            services.TryAddSingleton<IASRService>(sp => sp.GetRequiredService<IStreamASRService>());
            return services;
        }
    }
}
