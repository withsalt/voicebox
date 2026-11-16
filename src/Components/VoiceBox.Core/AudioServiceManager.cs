using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Diagnostics.CodeAnalysis;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.Core.Services.HPFService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;

namespace VoiceBox.Core
{
    public class AudioServiceManager : IAudioServiceManager
    {
        private ServiceCollection _services = new ServiceCollection();
        private ServiceProvider? _serviceProvider;
        private IServiceScope? _scope;

        private readonly object _lock = new object();
        private volatile bool _isDisposed = false;

        public AudioServiceManager()
        {
            // 默认注册一个 "无输出" 的 logging 基础设施 (没有任何 provider, 等价 NullLogger)。
            // 调用方可通过 AddLogger(...) 追加 provider / 配置最小级别。
            _services.AddLogging();
        }

        public ServiceRegistration<TService, TImplementation> AddService<TService, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TImplementation>(string? name = null)
            where TService : class
            where TImplementation : class, TService
        {
            ObjectDisposedException.ThrowIf(_isDisposed, this);
            return new ServiceRegistration<TService, TImplementation>(this, name);
        }

        #region AddService

        public AudioServiceManager AddService<TService, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TImplementation, TOptions>(Action<TOptions>? configure, string? name = null)
            where TService : class
            where TImplementation : class, TService
            where TOptions : class
        {
            ObjectDisposedException.ThrowIf(_isDisposed, this);

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_isDisposed, this);
                if (_serviceProvider != null)
                {
                    throw new InvalidOperationException("Cannot configure services after an instance has been requested.");
                }

                if (string.IsNullOrEmpty(name))
                {
                    if (_services.Any(s => s.ServiceType == typeof(TService) && s.ServiceKey is null))
                    {
                        throw new InvalidOperationException($"Default {typeof(TService).Name} is already registered.");
                    }

                    if (configure != null)
                        _services.Configure(configure);

                    _services.TryAddSingleton<TService, TImplementation>();
                }
                else
                {
                    if (_services.Any(s => s.ServiceType == typeof(TService) && name.Equals(s.ServiceKey)))
                    {
                        throw new InvalidOperationException($"{typeof(TService).Name} with name '{name}' is already registered.");
                    }

                    if (configure != null)
                        _services.Configure<TOptions>(name, configure);

                    _services.AddKeyedSingleton<TService>(name, (sp, _) =>
                    {
                        var optionsMonitor = sp.GetRequiredService<IOptionsMonitor<TOptions>>();
                        var namedOptions = optionsMonitor.Get(name);
                        var optionsWrapper = new OptionsWrapper<TOptions>(namedOptions);
                        return (TService)ActivatorUtilities.CreateInstance(sp, typeof(TImplementation), optionsWrapper);
                    });
                }
            }
            return this;
        }

        public AudioServiceManager AddService<TService>(Action<IServiceCollection> addAction) where TService : class
        {
            ObjectDisposedException.ThrowIf(_isDisposed, this);
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_isDisposed, this);
                // Enforce a "configure-then-use" pattern.
                if (_serviceProvider != null)
                {
                    throw new InvalidOperationException("Cannot configure services after an instance has been requested.");
                }
                addAction(_services);
            }
            return this;
        }

        /// <summary>
        /// 注册日志组件。未调用时, 容器内 <see cref="ILogger{T}"/> 解析为无输出的 NullLogger。
        /// </summary>
        /// <param name="configure">配置 <see cref="ILoggingBuilder"/>, 例如 AddSimpleConsole / SetMinimumLevel。</param>
        public AudioServiceManager AddLogger(Action<ILoggingBuilder> configure)
        {
            ArgumentNullException.ThrowIfNull(configure);
            ObjectDisposedException.ThrowIf(_isDisposed, this);
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_isDisposed, this);
                if (_serviceProvider != null)
                {
                    throw new InvalidOperationException("Cannot configure services after an instance has been requested.");
                }
                _services.AddLogging(configure);
            }
            return this;
        }

        /// <summary>
        /// 直接使用外部已构建好的 <see cref="ILoggerFactory"/> (例如宿主 DI 中的实例)。
        /// 会替换默认 LoggerFactory; 此后容器内所有 <see cref="ILogger{T}"/> 都从该 factory 创建。
        /// </summary>
        public AudioServiceManager AddLogger(ILoggerFactory loggerFactory)
        {
            ArgumentNullException.ThrowIfNull(loggerFactory);
            ObjectDisposedException.ThrowIf(_isDisposed, this);
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_isDisposed, this);
                if (_serviceProvider != null)
                {
                    throw new InvalidOperationException("Cannot configure services after an instance has been requested.");
                }
                _services.AddLogging();
                _services.Replace(ServiceDescriptor.Singleton(loggerFactory));
            }
            return this;
        }

        public AudioServiceManager AddServiceWithoutConfigure<TService, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)] TImplementation>(string? name = null)
            where TService : class
            where TImplementation : class, TService
        {
            ObjectDisposedException.ThrowIf(_isDisposed, this);
            return AddService<TService, TImplementation>(name).Configure<object>();
        }

        private TService GetInstance<TService>(string? name = null) where TService : notnull
        {
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_isDisposed, this);
                var scope = GetOrCreateScope();

                if (string.IsNullOrEmpty(name))
                {
                    if (!_services.Any(s => s.ServiceType == typeof(TService) && s.ServiceKey is null))
                    {
                        throw new InvalidOperationException($"{typeof(TService).Name} is not registered. Please call the corresponding AddService method before getting an instance.");
                    }

                    return scope.ServiceProvider.GetRequiredService<TService>();
                }
                else
                {
                    if (!_services.Any(s => s.ServiceType == typeof(TService) && name.Equals(s.ServiceKey)))
                    {
                        throw new InvalidOperationException($"{typeof(TService).Name} with name '{name}' is not registered. Please call the corresponding AddService method before getting an instance.");
                    }

                    return scope.ServiceProvider.GetRequiredKeyedService<TService>(name);
                }
            }
        }

        #endregion

        #region GetInstance

        public IAudioRecordService GetAudioRecordInstance(string? name = null) => GetInstance<IAudioRecordService>(name);

        public IAudioPlayService GetAudioPlayInstance(string? name = null) => GetInstance<IAudioPlayService>(name);

        public IAudioDuplexService GetAudioDuplexInstance(string? name = null) => GetInstance<IAudioDuplexService>(name);

        public IVoiceActivityDetectService GetVoiceActivityDetectInstance(string? name = null) => GetInstance<IVoiceActivityDetectService>(name);

        public IDigitalNoiseReductionService GetDNRServiceInstance(string? name = null) => GetInstance<IDigitalNoiseReductionService>(name);

        public IAcousticEchoCancellationService GetAECServiceInstance(string? name = null) => GetInstance<IAcousticEchoCancellationService>(name);

        public IAcousticProcessingMetricsSource GetAcousticProcessingMetricsSource(string? name = null)
        {
            IAcousticEchoCancellationService processor = GetAECServiceInstance(name);
            return processor as IAcousticProcessingMetricsSource
                ?? throw new InvalidOperationException(
                    $"The registered {nameof(IAcousticEchoCancellationService)} does not expose {nameof(IAcousticProcessingMetricsSource)}.");
        }

        public IHighPassFilterService GetHighPassFilterServiceInstance(string? name = null) => GetInstance<IHighPassFilterService>(name);

        public IStreamASRService GetStreamASRServiceInstance(string? name = null) => GetInstance<IStreamASRService>(name);

        public IASRService GetOfflineASRServiceInstance(string? name = null) => GetInstance<IASRService>(name);

        public IWakeupService GetWakeupServiceInstance(string? name = null) => GetInstance<IWakeupService>(name);

        public ISpeakerService GetSpeakerServiceInstance(string? name = null) => GetInstance<ISpeakerService>(name);

        public ITTSService GetTTSServiceInstance(string? name = null) => GetInstance<ITTSService>(name);

        public IStreamTTSService GetStreamTTSServiceInstance(string? name = null) => GetInstance<IStreamTTSService>(name);

        public ILLMService GetLLMServiceInstance(string? name = null) => GetInstance<ILLMService>(name);

        public IStreamLLMService GetStreamLLMServiceInstance(string? name = null) => GetInstance<IStreamLLMService>(name);

        #endregion

        public IAudioServiceManager Build()
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException(nameof(AudioServiceManager));
            }
            _ = GetOrCreateScope();
            return this;
        }

        private IServiceScope GetOrCreateScope()
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException(nameof(AudioServiceManager));
            }

            // Fast path check to avoid locking if already initialized.
            if (_scope != null)
            {
                return _scope;
            }

            lock (_lock)
            {
                if (_isDisposed)
                {
                    throw new ObjectDisposedException(nameof(AudioServiceManager));
                }

                // Double-checked locking.
                if (_scope != null)
                {
                    return _scope;
                }

                _serviceProvider = _services.BuildServiceProvider();
                _scope = _serviceProvider.CreateScope();
            }
            return _scope;
        }

        /// <summary>
        /// Disposes the service provider and all created services.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                lock (_lock)
                {
                    if (_isDisposed)
                    {
                        return;
                    }

                    _isDisposed = true;
                    // Child scopes are not disposed by disposing the root provider.
                    _scope?.Dispose();
                    _scope = null;
                    _serviceProvider?.Dispose();
                    _serviceProvider = null;
                    _services = null!;
                }
            }
            else
            {
                _isDisposed = true;
            }
        }
    }

}
