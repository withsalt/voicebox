using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Speaker.WeSpeaker.Models;

namespace VoiceBox.Speaker.WeSpeaker
{
    /// <summary>
    /// WeSpeaker 说话人验证服务的依赖注入扩展。
    /// </summary>
    public static class WeSpeakerServiceExtension
    {
        /// <summary>向 AudioServiceManager 注册 WeSpeaker 说话人验证服务。</summary>
        public static AudioServiceManager AddWeSpeakerService(
            this AudioServiceManager services,
            Action<WeSpeakerOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<ISpeakerService, WeSpeakerService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<ISpeakerService, WeSpeakerService>().Configure(configure);
            }
            return services;
        }

        /// <summary>向 IServiceCollection 注册 WeSpeaker 说话人验证服务 (单例)。</summary>
        public static IServiceCollection AddWeSpeakerService(
            this IServiceCollection services,
            Action<WeSpeakerOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            services.TryAddSingleton<ISpeakerService, WeSpeakerService>();
            return services;
        }
    }
}
