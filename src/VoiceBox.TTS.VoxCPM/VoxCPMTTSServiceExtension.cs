using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.TTSService;

namespace VoiceBox.TTS.VoxCPM
{
    public static class VoxCPMTTSServiceExtension
    {
        /// <summary>注册 VoxCPM TTS 服务 (同时注册 IStreamTTSService 与 ITTSService)。</summary>
        public static AudioServiceManager AddVoxCPMTTSService(
            this AudioServiceManager services, Action<VoxCPMTTSOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IStreamTTSService, VoxCPMTTSService>(serviceName).Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.AddKeyedSingleton<ITTSService>(serviceName, (sp, _) =>
                        sp.GetRequiredKeyedService<IStreamTTSService>(serviceName)));
            }
            else
            {
                services.AddService<IStreamTTSService, VoxCPMTTSService>().Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>()));
            }
            return services;
        }

        /// <summary>注册 VoxCPM TTS 服务到通用 DI 容器 (同时注册 IStreamTTSService 与 ITTSService)。</summary>
        public static IServiceCollection AddVoxCPMTTSService(
            this IServiceCollection services, Action<VoxCPMTTSOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            services.TryAddSingleton<IStreamTTSService, VoxCPMTTSService>();
            services.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>());
            return services;
        }
    }
}
