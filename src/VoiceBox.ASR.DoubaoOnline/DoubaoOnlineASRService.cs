using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.ASR.DoubaoOnline.Session;
using VoiceBox.ASR.DoubaoOnline.Transport;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.DoubaoOnline
{
    /// <summary>
    /// 基于火山引擎「豆包流式语音识别模型 2.0」双向流式 WebSocket 接口的在线 ASR 服务。
    /// </summary>
    /// <remarks>
    /// 架构
    /// ----
    ///   * <see cref="SeedAsrConnectionWorker"/> — 每会话一条 WebSocket，跑收发两个后台循环。
    ///   * <see cref="SeedAsrSession"/> — 每会话状态、音频 ingress channel、结果 channel、最终结果 TCS；
    ///     共享服务管理器保留会话，直到资源释放。
    ///   * <see cref="Transport.IWebSocketChannel"/> — 可替换的传输抽象 (生产用 ClientWebSocket，
    ///     测试用 fake)。
    ///
    /// 与本地 ONNX ASR 不同，这里没有共享推理闸门：并发能力取决于云端与本地连接数上限
    /// (<see cref="DoubaoOnlineASROptions.MaxConcurrentSessions"/>)。
    /// </remarks>
    public sealed class DoubaoOnlineASRService : BaseASRService, IStreamASRService
    {
        private readonly DoubaoOnlineASROptions _options;
        private readonly ILogger<DoubaoOnlineASRService> _logger;
        private readonly IWebSocketChannelFactory _channelFactory;
        public DoubaoOnlineASRService(
            IOptions<DoubaoOnlineASROptions> options,
            ILogger<DoubaoOnlineASRService>? logger = null)
            : this(options, channelFactory: null, logger) { }

        // 供测试注入 fake 传输工厂的构造。
        internal DoubaoOnlineASRService(
            IOptions<DoubaoOnlineASROptions> options,
            IWebSocketChannelFactory? channelFactory,
            ILogger<DoubaoOnlineASRService>? logger = null)
            : base(GetMaxConcurrentSessions(options))
        {
            ArgumentNullException.ThrowIfNull(options);
            _options = options.Value ?? throw new ArgumentNullException(nameof(options));
            _logger = logger ?? NullLogger<DoubaoOnlineASRService>.Instance;
            _channelFactory = channelFactory ?? new ClientWebSocketChannelFactory();

            ValidateOptions(_options);
        }

        protected override AsrSession CreateSession(string sessionId)
        {
            string requestId = Guid.NewGuid().ToString();
            string connectId = Guid.NewGuid().ToString();
            IWebSocketChannel? channel = null;
            try
            {
                channel = _channelFactory.Create(SeedAsrRequestBuilder.BuildUri(_options),
                    SeedAsrRequestBuilder.BuildHeaders(_options, requestId, connectId), _options.ConnectTimeoutSec);
                return new SeedAsrSession(sessionId, requestId, connectId,
                    ComputeMaxBufferedSamples(_options), channel, _options,
                    SeedAsrRequestBuilder.BuildRequestJson(_options), _logger);
            }
            catch
            {
                channel?.Dispose();
                throw;
            }
        }

        public async IAsyncEnumerable<AsrResult> GetStreamResultAsync(
            string sessionId,
            [EnumeratorCancellation] CancellationToken ct = default)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
            EnsureServiceActive();

            if (!TryGetSession<SeedAsrSession>(sessionId, out var session))
                throw new InvalidOperationException($"Session not found: {sessionId}");

            await foreach (var item in session.ResultReader.ReadAllAsync(ct).ConfigureAwait(false))
                yield return item;
        }

        private static void ValidateOptions(DoubaoOnlineASROptions o)
        {
            if (string.IsNullOrWhiteSpace(o.ApiKey))
                throw new ArgumentException(
                    "Authentication is not configured: set ApiKey.");
            if (string.IsNullOrWhiteSpace(o.ResourceId))
                throw new ArgumentException("DoubaoOnlineASROptions.ResourceId must be set.");
            if (string.IsNullOrWhiteSpace(o.Host))
                throw new ArgumentException("DoubaoOnlineASROptions.Host must be set.");
            if (o.SampleRate != 16000)
                throw new ArgumentOutOfRangeException(nameof(o), "SampleRate must be 16000 (the only supported rate).");
            if (o.Channels is not (1 or 2))
                throw new ArgumentOutOfRangeException(nameof(o), "Channels must be 1 or 2.");
            if (o.PackageDurationMs <= 0)
                throw new ArgumentOutOfRangeException(nameof(o), "PackageDurationMs must be > 0.");
            if ((long)o.SampleRate * o.Channels * o.PackageDurationMs / 1000 > int.MaxValue / 2)
                throw new ArgumentOutOfRangeException(nameof(o), "PackageDurationMs exceeds the supported audio frame buffer size.");
            if (o.MaxConcurrentSessions < 1)
                throw new ArgumentOutOfRangeException(nameof(o), "MaxConcurrentSessions must be ≥ 1.");
            if (o.MaxBufferChunks < 1)
                throw new ArgumentOutOfRangeException(nameof(o), "MaxBufferChunks must be ≥ 1.");
            if (o.MaxBufferedAudioMs < 1)
                throw new ArgumentOutOfRangeException(nameof(o), "MaxBufferedAudioMs must be ≥ 1.");
        }

        private static long ComputeMaxBufferedSamples(DoubaoOnlineASROptions options)
        {
            long configuredMs = Math.Min(
                options.MaxBufferedAudioMs,
                checked((long)options.MaxBufferChunks * options.PackageDurationMs));
            return Math.Max(
                1,
                checked(configuredMs * options.SampleRate * options.Channels / 1000));
        }

        private static int GetMaxConcurrentSessions(IOptions<DoubaoOnlineASROptions> options)
        {
            ArgumentNullException.ThrowIfNull(options);
            return options.Value?.MaxConcurrentSessions
                ?? throw new ArgumentException("DoubaoOnlineASROptions cannot be null.", nameof(options));
        }
    }
}
