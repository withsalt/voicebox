using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using VoiceBox.Core;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.LLM.OnnxGenAI
{
    public static class OnnxGenAILLMServiceExtension
    {
        /// <summary>
        /// 将 <see cref="OnnxGenAILLMService"/> 注册到 <see cref="IServiceCollection"/>。
        /// 同时注册 <see cref="ILLMService"/> 与 <see cref="IStreamLLMService"/>, 二者共享同一实例。
        /// </summary>
        public static IServiceCollection AddOnnxGenAILLMService(this IServiceCollection services, Action<OnnxGenAILLMOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            // 不变式校验集中在 OnnxGenAILLMOptionsValidator (与直接构造服务共用同一套规则);
            // 目录存在性单独在此校验, 以便启动即失败并给出清晰信息。
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IValidateOptions<OnnxGenAILLMOptions>, OnnxGenAILLMOptionsValidator>());
            services.AddOptions<OnnxGenAILLMOptions>()
                .Configure(configure)
                .Validate(o => System.IO.Directory.Exists(o.ModelPath), "OnnxGenAILLMOptions.ModelPath does not exist.")
                .ValidateOnStart();

            services.TryAddSingleton<OnnxGenAILLMService>();
            services.TryAddSingleton<IStreamLLMService>(sp => sp.GetRequiredService<OnnxGenAILLMService>());
            services.TryAddSingleton<ILLMService>(sp => sp.GetRequiredService<OnnxGenAILLMService>());
            return services;
        }

        /// <summary>
        /// 注册到 <see cref="AudioServiceManager"/>, 与 Qwen3/Kokoro 等服务保持调用风格一致。
        /// </summary>
        public static AudioServiceManager AddOnnxGenAILLMService(this AudioServiceManager manager, Action<OnnxGenAILLMOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(manager);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                manager.AddService<IStreamLLMService, OnnxGenAILLMService>(serviceName).Configure(configure);
                manager.AddService<ILLMService, OnnxGenAILLMService>(serviceName).Configure(configure);
            }
            else
            {
                manager.AddService<IStreamLLMService, OnnxGenAILLMService>().Configure(configure);
                manager.AddService<ILLMService, OnnxGenAILLMService>().Configure(configure);
            }
            return manager;
        }
    }
}
