using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using VoiceBox.Core;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.LLM.LLamaSharp
{
    public static class LLamaSharpLLMServiceExtension
    {
        /// <summary>
        /// 注册一个自定义模型兼容适配器。适配器会优先于内置自适应模板参与匹配；
        /// 适合应用级复用。单个服务的特例也可放进
        /// <see cref="LLamaSharpLLMOptions.AdditionalModelProfiles"/>。
        /// </summary>
        public static IServiceCollection AddLLamaSharpModelProfile<TProfile>(this IServiceCollection services)
            where TProfile : class, ILlamaModelProfile
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddEnumerable(ServiceDescriptor.Singleton<ILlamaModelProfile, TProfile>());
            return services;
        }

        /// <summary>
        /// 将 <see cref="LLamaSharpLLMService"/> 注册到 <see cref="IServiceCollection"/>。
        /// 注册 <see cref="ILLMService"/>, <see cref="IStreamLLMService"/> 和
        /// <see cref="IToolCallingStreamLLMService"/>, 全部共享同一实例。
        /// </summary>
        public static IServiceCollection AddLLamaSharpLLMService(this IServiceCollection services, Action<LLamaSharpLLMOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            // 不变式校验集中在 LLamaSharpLLMOptionsValidator (与直接构造服务共用同一套规则);
            // 文件存在性单独在此校验, 以便启动即失败并给出清晰信息。
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IValidateOptions<LLamaSharpLLMOptions>, LLamaSharpLLMOptionsValidator>());
            services.AddOptions<LLamaSharpLLMOptions>()
                .Configure(configure)
                .Validate(o => System.IO.File.Exists(o.ModelPath), "LLamaSharpLLMOptions.ModelPath does not exist.")
                .ValidateOnStart();

            services.TryAddSingleton<LLamaSharpLLMService>();
            services.TryAddSingleton<IStreamLLMService>(sp => sp.GetRequiredService<LLamaSharpLLMService>());
            services.TryAddSingleton<ILLMService>(sp => sp.GetRequiredService<LLamaSharpLLMService>());
            services.TryAddSingleton<IToolCallingStreamLLMService>(sp => sp.GetRequiredService<LLamaSharpLLMService>());
            return services;
        }

        /// <summary>
        /// 注册到 <see cref="AudioServiceManager"/>, 与 OnnxGenAI/Kokoro 等服务保持调用风格一致。
        /// </summary>
        public static AudioServiceManager AddLLamaSharpLLMService(this AudioServiceManager manager, Action<LLamaSharpLLMOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(manager);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                manager.AddService<IStreamLLMService, LLamaSharpLLMService>(serviceName).Configure(configure);
                manager.AddService<ILLMService, LLamaSharpLLMService>(serviceName).Configure(configure);
            }
            else
            {
                manager.AddService<IStreamLLMService, LLamaSharpLLMService>().Configure(configure);
                manager.AddService<ILLMService, LLamaSharpLLMService>().Configure(configure);
            }
            return manager;
        }
    }
}
