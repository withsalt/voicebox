using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.TTSService;

namespace VoiceBox.TTS.DoubaoOnline
{
    /// <summary>
    /// <see cref="DoubaoOnlineTTSService"/> 的注册扩展。既支持 VoiceBox 的
    /// <see cref="AudioServiceManager"/>，也支持标准 <see cref="IServiceCollection"/>。
    /// 同时把服务登记为 <see cref="IStreamTTSService"/> 与 <see cref="ITTSService"/>。
    /// </summary>
    public static class DoubaoOnlineTTSServiceExtension
    {
        public static AudioServiceManager AddDoubaoOnlineTTSService(
            this AudioServiceManager services,
            Action<DoubaoOnlineTTSOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IStreamTTSService, DoubaoOnlineTTSService>(serviceName).Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.AddKeyedSingleton<ITTSService>(serviceName, (sp, _) =>
                        sp.GetRequiredKeyedService<IStreamTTSService>(serviceName)));
            }
            else
            {
                services.AddService<IStreamTTSService, DoubaoOnlineTTSService>().Configure(configure);
                services.AddService<ITTSService>(sc =>
                    sc.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>()));
            }
            return services;
        }

        public static IServiceCollection AddDoubaoOnlineTTSService(
            this IServiceCollection services,
            Action<DoubaoOnlineTTSOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            services.Configure(configure);
            services.TryAddSingleton<IStreamTTSService, DoubaoOnlineTTSService>();
            services.TryAddSingleton<ITTSService>(sp => sp.GetRequiredService<IStreamTTSService>());
            return services;
        }
    }
}
