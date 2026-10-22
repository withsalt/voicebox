using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Wakeup.OpenWakeWord.Models;

namespace VoiceBox.Wakeup.OpenWakeWord
{
    public static class OpenWakeWordServiceExtension
    {
        /// <summary>
        /// 注册 OpenWakeWord 唤醒词检测服务到 <see cref="AudioServiceManager"/>。
        /// </summary>
        public static AudioServiceManager AddOpenWakeWordService(
            this AudioServiceManager services,
            Action<OpenWakeWordOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IWakeupService, OpenWakeWordService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IWakeupService, OpenWakeWordService>().Configure(configure);
            }
            return services;
        }

        /// <summary>
        /// 注册 OpenWakeWord 到通用 <see cref="IServiceCollection"/> (非 AudioServiceManager 场景)。
        /// </summary>
        public static IServiceCollection AddOpenWakeWordService(
            this IServiceCollection services,
            Action<OpenWakeWordOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            services.Configure(configure);
            services.TryAddSingleton<IWakeupService, OpenWakeWordService>();
            return services;
        }
    }
}
