using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Wakeup.FsmnKws.Models;

namespace VoiceBox.Wakeup.FsmnKws
{
    public static class FsmnKwsServiceExtension
    {
        /// <summary>
        /// 注册 FsmnKws (小云 FSMN+CTC) 唤醒服务到 <see cref="AudioServiceManager"/>。
        /// </summary>
        public static AudioServiceManager AddFsmnKwsService(
            this AudioServiceManager services,
            Action<FsmnKwsOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IWakeupService, FsmnKwsService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IWakeupService, FsmnKwsService>().Configure(configure);
            }
            return services;
        }

        /// <summary>
        /// 注册到通用 <see cref="IServiceCollection"/> (非 AudioServiceManager 场景)。
        /// </summary>
        public static IServiceCollection AddFsmnKwsService(
            this IServiceCollection services,
            Action<FsmnKwsOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            services.Configure(configure);
            services.TryAddSingleton<IWakeupService, FsmnKwsService>();
            return services;
        }
    }
}
