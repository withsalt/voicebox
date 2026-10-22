using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Speaker.CampPlus.Models;

namespace VoiceBox.Speaker.CampPlus
{
    /// <summary>
    /// CAM++ 说话人验证服务的依赖注入扩展。注册形式与 Hush/Silero/FsmnKws 等其它服务保持一致。
    /// </summary>
    public static class CampPlusSpeakerServiceExtension
    {
        /// <summary>向 AudioServiceManager 注册 CAM++ 说话人验证服务。</summary>
        public static AudioServiceManager AddCampPlusSpeakerService(
            this AudioServiceManager services,
            Action<CampPlusOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<ISpeakerService, CampPlusSpeakerService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<ISpeakerService, CampPlusSpeakerService>().Configure(configure);
            }
            return services;
        }

        /// <summary>向 IServiceCollection 注册 CAM++ 说话人验证服务 (单例)。</summary>
        public static IServiceCollection AddCampPlusSpeakerService(
            this IServiceCollection services,
            Action<CampPlusOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            services.TryAddSingleton<ISpeakerService, CampPlusSpeakerService>();
            return services;
        }
    }
}
