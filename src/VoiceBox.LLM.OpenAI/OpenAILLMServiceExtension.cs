using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using VoiceBox.Core;
using VoiceBox.Core.Services.LLMService;

namespace VoiceBox.LLM.OpenAI
{
    public static class OpenAILLMServiceExtension
    {
        /// <summary>
        /// 将 <see cref="OpenAILLMService"/> 注册到 <see cref="IServiceCollection"/>。
        /// 同时注册 <see cref="ILLMService"/> 与 <see cref="IStreamLLMService"/>, 二者共享同一实例。
        /// </summary>
        public static IServiceCollection AddOpenAILLMService(this IServiceCollection services, Action<OpenAILLMOptions> configure)
        {
            ArgumentNullException.ThrowIfNull(services);
            ArgumentNullException.ThrowIfNull(configure);

            services.AddOptions<OpenAILLMOptions>()
                .Configure(configure)
                .Validate(o => !string.IsNullOrWhiteSpace(o.ApiKey), "OpenAILLMOptions.ApiKey cannot be null or empty.")
                .Validate(o => o.Endpoint is not null, "OpenAILLMOptions.Endpoint cannot be null.")
                .Validate(o => !string.IsNullOrWhiteSpace(o.Model), "OpenAILLMOptions.Model cannot be null or empty.")
                .Validate(o => o.MaxConcurrentRequests >= 1, "OpenAILLMOptions.MaxConcurrentRequests must be >= 1.")
                .Validate(o => o.StreamChannelCapacity >= 0, "OpenAILLMOptions.StreamChannelCapacity must be >= 0.")
                .Validate(o => o.HttpTimeout > TimeSpan.Zero, "OpenAILLMOptions.HttpTimeout must be > 0.");

            services.TryAddSingleton<OpenAILLMService>();
            services.TryAddSingleton<IStreamLLMService>(sp => sp.GetRequiredService<OpenAILLMService>());
            services.TryAddSingleton<ILLMService>(sp => sp.GetRequiredService<OpenAILLMService>());
            return services;
        }

        /// <summary>
        /// 注册到 <see cref="AudioServiceManager"/>, 与其他 VoiceBox 服务保持调用风格一致。
        /// </summary>
        public static AudioServiceManager AddOpenAILLMService(this AudioServiceManager manager, Action<OpenAILLMOptions> configure, string? serviceName = "")
        {
            ArgumentNullException.ThrowIfNull(manager);
            ArgumentNullException.ThrowIfNull(configure);

            if (!string.IsNullOrWhiteSpace(serviceName))
            {
                manager.AddService<IStreamLLMService, OpenAILLMService>(serviceName).Configure(configure);
                manager.AddService<ILLMService, OpenAILLMService>(serviceName).Configure(configure);
            }
            else
            {
                manager.AddService<IStreamLLMService, OpenAILLMService>().Configure(configure);
                manager.AddService<ILLMService, OpenAILLMService>().Configure(configure);
            }
            return manager;
        }
    }
}
