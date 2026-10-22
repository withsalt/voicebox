using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.DoubaoOnline.Protocol;

namespace VoiceBox.TTS.DoubaoOnline.Session
{
    /// <summary>
    /// 单次流式合成会话的状态。持有：
    ///   * 下行音频 channel (接收 loop 生产, GetStreamAudioAsync 消费)；
    ///   * 三个握手门 (ConnectionStarted / SessionStarted / SessionFinished) 供 worker 推进流程；
    ///   * 字幕与计费信息 (若开启)。
    /// 单生产者 (接收 loop) / 单消费者 (合成流) 音频。
    /// </summary>
    internal sealed class SeedTtsSession : IDisposable
    {
        private readonly Channel<TtsAudioChunk> _audioChannel;
        private readonly TaskCompletionSource _connectionStarted =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource _sessionStarted =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource _sessionFinished =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly CancellationTokenSource _cts = new();
        private readonly ILogger _logger;

        private readonly List<WordTimestamp> _subtitles = new();
        private long _emittedSamples;
        private int _disposed;

        public string SessionId { get; }
        public string RequestId { get; }
        public string ConnectId { get; }
        public DateTime CreatedAt { get; }
        public int SampleRate { get; }
        public int Channels { get; }

        public ChannelReader<TtsAudioChunk> AudioReader => _audioChannel.Reader;
        public CancellationToken CancellationToken => _cts.Token;

        /// <summary>建连成功后完成 (收到 ConnectionStarted)。</summary>
        public Task ConnectionStarted => _connectionStarted.Task;

        /// <summary>会话开始后完成 (收到 SessionStarted)。</summary>
        public Task SessionStarted => _sessionStarted.Task;

        /// <summary>会话结束后完成 (收到 SessionFinished)。</summary>
        public Task SessionFinished => _sessionFinished.Task;

        /// <summary>握手后记录的服务端 logid，便于日志排障。</summary>
        public string? LogId { get; set; }

        /// <summary>本次合成的计费字数 (开启计费返回时)。</summary>
        public int? BilledTextWords { get; private set; }

        /// <summary>字幕 (字级别时间戳) 快照。</summary>
        public IReadOnlyList<WordTimestamp> Subtitles => _subtitles;

        public SeedTtsSession(
            string sessionId,
            string requestId,
            string connectId,
            int sampleRate,
            int channels,
            ILogger? logger = null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
            SessionId = sessionId;
            RequestId = requestId;
            ConnectId = connectId;
            SampleRate = sampleRate;
            Channels = channels;
            CreatedAt = DateTime.UtcNow;
            _logger = logger ?? NullLogger.Instance;

            // 兜底观察握手门的异常: 正常路径 StartAsync 会 await 三个门, 已被观察; 但若失败发生在
            // 无人 await 的时刻 (例如仅消费音频流), faulted task 会在 GC 时触发 UnobservedTaskException。
            ObserveFault(_connectionStarted.Task);
            ObserveFault(_sessionStarted.Task);
            ObserveFault(_sessionFinished.Task);

            _audioChannel = Channel.CreateUnbounded<TtsAudioChunk>(new UnboundedChannelOptions
            {
                SingleReader = true,
                SingleWriter = true,
            });
        }

        private static void ObserveFault(Task t) => _ = t.ContinueWith(
            static x => _ = x.Exception,
            CancellationToken.None,
            TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);

        /// <summary>把一段解码后的 PCM float 采样作为音频块推给消费者。</summary>
        public void PublishAudio(float[] samples)
        {
            if (samples.Length == 0) return;
            _emittedSamples += samples.Length;
            _audioChannel.Writer.TryWrite(new TtsAudioChunk
            {
                Samples = samples,
                SampleRate = SampleRate,
                Channels = Channels,
                EmittedSeconds = (double)_emittedSamples / Math.Max(1, SampleRate * Channels),
                IsFinal = false,
            });
        }

        public void AddSubtitles(IEnumerable<WordTimestamp> words) => _subtitles.AddRange(words);

        public void SetUsage(int textWords) => BilledTextWords = textWords;

        public void MarkConnectionStarted() => _connectionStarted.TrySetResult();

        public void MarkSessionStarted() => _sessionStarted.TrySetResult();

        /// <summary>会话正常结束：补发一个空的 IsFinal 块并 complete 音频写端。</summary>
        public void MarkSessionFinished()
        {
            _connectionStarted.TrySetResult();
            _sessionStarted.TrySetResult();
            _audioChannel.Writer.TryWrite(new TtsAudioChunk
            {
                Samples = Array.Empty<float>(),
                SampleRate = SampleRate,
                Channels = Channels,
                EmittedSeconds = (double)_emittedSamples / Math.Max(1, SampleRate * Channels),
                IsFinal = true,
            });
            _sessionFinished.TrySetResult();
            _audioChannel.Writer.TryComplete();
        }

        /// <summary>以异常收尾 (幂等)：所有握手门与音频流都以该异常收尾。</summary>
        public void TrySetError(Exception ex)
        {
            _connectionStarted.TrySetException(ex);
            _sessionStarted.TrySetException(ex);
            _sessionFinished.TrySetException(ex);
            _audioChannel.Writer.TryComplete(ex);
        }

        public void Cancel()
        {
            _connectionStarted.TrySetCanceled();
            _sessionStarted.TrySetCanceled();
            _sessionFinished.TrySetCanceled();
            try { _cts.Cancel(); } catch { }
            _audioChannel.Writer.TryComplete();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            Cancel();
            _cts.Dispose();
        }
    }
}
