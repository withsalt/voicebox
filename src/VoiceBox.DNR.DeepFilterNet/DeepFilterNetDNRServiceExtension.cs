using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.DNR.DeepFilterNet.Models;

namespace VoiceBox.DNR.DeepFilterNet
{
    /// <summary>
    /// DeepFilterNet 降噪服务的依赖注入扩展。注册形式与 DPDFNet/RNNoise 等其它降噪服务保持一致。
    /// </summary>
    public static class DeepFilterNetDNRServiceExtension
    {
        /// <summary>
        /// 向 <see cref="AudioServiceManager"/> 注册 DeepFilterNet 降噪服务。
        /// </summary>
        /// <param name="services">音频服务管理器。</param>
        /// <param name="configure">配置回调。</param>
        /// <param name="serviceName">可选命名实例。为空时注册为默认实例。</param>
        public static AudioServiceManager AddDeepFilterNetDNRService(this AudioServiceManager services, Action<DeepFilterNetOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IDigitalNoiseReductionService, DeepFilterNetDNRService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IDigitalNoiseReductionService, DeepFilterNetDNRService>().Configure(configure);
            }
            return services;
        }

        /// <summary>
        /// 向 <see cref="IServiceCollection"/> 注册 DeepFilterNet 降噪服务 (单例)。
        /// </summary>
        public static IServiceCollection AddDeepFilterNetDNRService(this IServiceCollection services, Action<DeepFilterNetOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            services.TryAddSingleton<IDigitalNoiseReductionService, DeepFilterNetDNRService>();
            return services;
        }
    }
}
