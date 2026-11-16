using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using VoiceBox.Audio.PortAudio.Models;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService;

namespace VoiceBox.Audio.PortAudio
{
    public static class PortAudioServiceExtension
    {
        public static AudioServiceManager AddPortAudioRecordService(this AudioServiceManager services, Action<PortAudioRecordOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IAudioRecordService, PortAudioRecordService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IAudioRecordService, PortAudioRecordService>().Configure(configure);
            }
            return services;
        }

        public static AudioServiceManager AddPortAudioPlayService(this AudioServiceManager services, Action<PortAudioPlayOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                services.AddService<IAudioPlayService, PortAudioPlayService>(serviceName).Configure(configure);
            }
            else
            {
                services.AddService<IAudioPlayService, PortAudioPlayService>().Configure(configure);
            }
            return services;
        }

        public static IServiceCollection AddPortAudioRecordService(this IServiceCollection services, Action<PortAudioRecordOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IAudioRecordService, PortAudioRecordService>();
            return services;
        }

        public static IServiceCollection AddPortAudioPlayService(this IServiceCollection services, Action<PortAudioPlayOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IAudioPlayService, PortAudioPlayService>();
            return services;
        }

        public static AudioServiceManager AddPortAudioDuplexService(this AudioServiceManager services, Action<PortAudioDuplexOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            // 使用 AddService 的 Action<IServiceCollection> 重载来确保 record/play/duplex 指向同一个实例。
            return services.AddService<IAudioRecordService>(sc =>
            {
                if (string.IsNullOrWhiteSpace(serviceName))
                {
                    sc.Configure(configure);
                    sc.TryAddSingleton<IAudioDuplexService, PortAudioDuplexService>();
                    sc.TryAddSingleton<IAudioRecordService>(sp => sp.GetRequiredService<IAudioDuplexService>());
                    sc.TryAddSingleton<IAudioPlayService>(sp => sp.GetRequiredService<IAudioDuplexService>());
                    return;
                }

                sc.Configure(serviceName, configure);
                sc.AddKeyedSingleton<IAudioDuplexService>(serviceName, (sp, _) =>
                {
                    var optionsMonitor = sp.GetRequiredService<IOptionsMonitor<PortAudioDuplexOptions>>();
                    var namedOptions = optionsMonitor.Get(serviceName);
                    return ActivatorUtilities.CreateInstance<PortAudioDuplexService>(
                        sp,
                        new OptionsWrapper<PortAudioDuplexOptions>(namedOptions));
                });
                sc.AddKeyedSingleton<IAudioRecordService>(serviceName, (sp, _) =>
                    sp.GetRequiredKeyedService<IAudioDuplexService>(serviceName));
                sc.AddKeyedSingleton<IAudioPlayService>(serviceName, (sp, _) =>
                    sp.GetRequiredKeyedService<IAudioDuplexService>(serviceName));
            });
        }

        public static IServiceCollection AddPortAudioFullDuplexService(this IServiceCollection services, Action<PortAudioDuplexOptions> configure)
        {
            services.Configure(configure);
            services.TryAddSingleton<IAudioDuplexService, PortAudioDuplexService>();
            services.TryAddSingleton<IAudioRecordService>(sp => sp.GetRequiredService<IAudioDuplexService>());
            services.TryAddSingleton<IAudioPlayService>(sp => sp.GetRequiredService<IAudioDuplexService>());
            return services;
        }
    }
}
