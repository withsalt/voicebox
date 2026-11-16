using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.DNR.Hush.Models;

namespace VoiceBox.DNR.Hush
{
    /// <summary>
    /// Hush 降噪服务的依赖注入扩展。注册形式与 DeepFilterNet/DPDFNet/RNNoise 等其它降噪服务保持一致。
    /// </summary>
    public static class HushDNRServiceExtension
    {
        /// <summary>
        /// 向 <see cref="AudioServiceManager"/> 注册 Hush 降噪服务。
        /// </summary>
        /// <param name="services">音频服务管理器。</param>
        /// <param name="configure">配置回调。</param>
        /// <param name="serviceName">可选命名实例。为空时注册为默认实例。</param>
        public static AudioServiceManager AddHushDNRService(this AudioServiceManager services, Action<HushOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IDigitalNoiseReductionService, HushDNRService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IDigitalNoiseReductionService, HushDNRService>().Configure(configure);
            }
            return services;
        }

        /// <summary>
        /// 向 <see cref="IServiceCollection"/> 注册 Hush 降噪服务 (单例)。
        /// </summary>
        public static IServiceCollection AddHushDNRService(this IServiceCollection services, Action<HushOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            services.TryAddSingleton<IDigitalNoiseReductionService, HushDNRService>();
            return services;
        }
    }
}
