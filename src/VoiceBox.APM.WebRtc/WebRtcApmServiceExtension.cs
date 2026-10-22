using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.APM.WebRtc.Models;
using VoiceBox.Core;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;

namespace VoiceBox.APM.WebRtc
{
    /// <summary>
    /// WebRTC M151 统一采集侧 APM 处理器的依赖注入扩展。
    /// 以 <see cref="IAcousticEchoCancellationService"/> 形式注册，直接接入
    /// <c>PortAudioDuplexService</c> 现有的采集处理钩子(拥有对齐的远端参考)。
    /// 启用本服务时应关闭独立的 HPF/DNR(它们已被 APM 一并完成)。
    /// </summary>
    public static class WebRtcApmServiceExtension
    {
        /// <summary>注册 WebRTC APM 统一处理器到 <see cref="AudioServiceManager"/>。</summary>
        public static AudioServiceManager AddWebRtcApmService(
            this AudioServiceManager services,
            Action<WebRtcApmOptions> configure,
            string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IAcousticEchoCancellationService, WebRtcApmService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IAcousticEchoCancellationService, WebRtcApmService>().Configure(configure);
            }
            return services;
        }

        /// <summary>注册 WebRTC APM 统一处理器到通用 <see cref="IServiceCollection"/>。</summary>
        public static IServiceCollection AddWebRtcApmService(
            this IServiceCollection services,
            Action<WebRtcApmOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.Configure(configure);
            services.AddWebRtcAec3DelayEstimator();
            bool processorAlreadyRegistered = services.Any(descriptor =>
                descriptor.ServiceType == typeof(IAcousticEchoCancellationService) &&
                descriptor.ServiceKey is null);
            if (!processorAlreadyRegistered)
            {
                services.AddSingleton<IAcousticEchoCancellationService, WebRtcApmService>();
                services.TryAddSingleton<IAcousticProcessingMetricsSource>(provider =>
                    (IAcousticProcessingMetricsSource)provider.GetRequiredService<IAcousticEchoCancellationService>());
            }
            return services;
        }

        /// <summary>
        /// 仅注册 WebRTC AEC3 声学路径延迟估计器工厂，不注册完整 APM 处理器。
        /// </summary>
        public static IServiceCollection AddWebRtcAec3DelayEstimator(
            this IServiceCollection services)
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddSingleton<
                IAcousticPathDelayEstimatorFactory,
                WebRtcAec3DelayEstimatorFactory>();
            return services;
        }
    }
}
