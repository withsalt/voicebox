using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.TTSService;

namespace VoiceBox.TTS.Qwen3
{
    public static class Qwen3TTSServiceExtension
    {
        /// <summary>注册 Qwen3-TTS 服务 (同时注册 IStreamTTSService 与 ITTSService)。</summary>
        public static AudioServiceManager AddQwen3TTSService(
            this AudioServiceManager services, Action<Qwen3TTSOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IStreamTTSService, Qwen3TTSService>(serviceName).Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.AddKeyedSingleton<ITTSService>(serviceName, (sp, _) =>
                        sp.GetRequiredKeyedService<IStreamTTSService>(serviceName)));
            }
            else
            {
                services.AddService<IStreamTTSService, Qwen3TTSService>().Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>()));
            }
            return services;
        }

        /// <summary>注册 Qwen3-TTS 服务到通用 DI 容器 (同时注册 IStreamTTSService 与 ITTSService)。</summary>
        public static IServiceCollection AddQwen3TTSService(
            this IServiceCollection services, Action<Qwen3TTSOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            services.TryAddSingleton<IStreamTTSService, Qwen3TTSService>();
            services.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>());
            return services;
        }
    }
}
