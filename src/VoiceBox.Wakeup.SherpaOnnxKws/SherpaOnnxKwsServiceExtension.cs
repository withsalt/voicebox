using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Wakeup.SherpaOnnxKws.Models;

namespace VoiceBox.Wakeup.SherpaOnnxKws
{
    public static class SherpaOnnxKwsServiceExtension
    {
        /// <summary>
        /// 注册 sherpa-onnx KWS 唤醒服务到 <see cref="AudioServiceManager"/>。
        /// </summary>
        public static AudioServiceManager AddSherpaOnnxKwsService(
            this AudioServiceManager services,
            Action<SherpaOnnxKwsOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IWakeupService, SherpaOnnxKwsService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IWakeupService, SherpaOnnxKwsService>().Configure(configure);
            }
            return services;
        }

        /// <summary>
        /// 注册到通用 <see cref="IServiceCollection"/> (非 AudioServiceManager 场景)。
        /// </summary>
        public static IServiceCollection AddSherpaOnnxKwsService(
            this IServiceCollection services,
            Action<SherpaOnnxKwsOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);
            services.Configure(configure);
            services.TryAddSingleton<IWakeupService, SherpaOnnxKwsService>();
            return services;
        }
    }
}
