using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.ASRService;

namespace VoiceBox.ASR.DoubaoOnline
{
    /// <summary>
    /// <see cref="DoubaoOnlineASRService"/> 的注册扩展。既支持 VoiceBox 的
    /// <see cref="AudioServiceManager"/>，也支持标准 <see cref="IServiceCollection"/>。
    /// 同时把服务登记为 <see cref="IStreamASRService"/> 与 <see cref="IASRService"/>。
    /// </summary>
    public static class DoubaoOnlineASRServiceExtension
    {
        public static AudioServiceManager AddDoubaoOnlineASRService(
            this AudioServiceManager services,
            Action<DoubaoOnlineASROptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IStreamASRService, DoubaoOnlineASRService>(serviceName).Configure(configure);
                services.AddService<IASRService>(sc =>
                    sc.AddKeyedSingleton<IASRService>(serviceName, (sp, _) =>
                        sp.GetRequiredKeyedService<IStreamASRService>(serviceName)));
            }
            else
            {
                services.AddService<IStreamASRService, DoubaoOnlineASRService>().Configure(configure);
                services.AddService<IASRService>(sc =>
                    sc.TryAddSingleton<IASRService>(sp => sp.GetRequiredService<IStreamASRService>()));
            }
            return services;
        }

        public static IServiceCollection AddDoubaoOnlineASRService(
            this IServiceCollection services,
            Action<DoubaoOnlineASROptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            services.Configure(configure);
            services.TryAddSingleton<IStreamASRService, DoubaoOnlineASRService>();
            services.TryAddSingleton<IASRService>(sp => sp.GetRequiredService<IStreamASRService>());
            return services;
        }
    }
}
