using System.Diagnostics;
using System.Runtime.CompilerServices;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal sealed class BenchmarkTimeProvider : TimeProvider
{
    private readonly object _timerSync = new();
    private readonly HashSet<BenchmarkTimer> _timers = [];
    private long _timestamp;
    private DateTimeOffset _utcNow;

    public BenchmarkTimeProvider()
    {
        Reset();
    }

    public override long TimestampFrequency => Stopwatch.Frequency;

    public override long GetTimestamp() => Volatile.Read(ref _timestamp);

    public override DateTimeOffset GetUtcNow() => _utcNow;

    public override ITimer CreateTimer(
        TimerCallback callback,
        object? state,
        TimeSpan dueTime,
        TimeSpan period)
    {
        ArgumentNullException.ThrowIfNull(callback);
        var timer = new BenchmarkTimer(this, callback, state);
        if (!timer.Change(dueTime, period))
            throw new ObjectDisposedException(nameof(BenchmarkTimer));
        return timer;
    }

    public void Reset()
    {
        Volatile.Write(ref _timestamp, Stopwatch.GetTimestamp());
        _utcNow = DateTimeOffset.UtcNow;
    }

    public void AdvanceSamples(int interleavedSamples, int sampleRate, int channels)
    {
        if (interleavedSamples < 0)
            throw new ArgumentOutOfRangeException(nameof(interleavedSamples));
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (channels <= 0 || interleavedSamples % channels != 0)
            throw new ArgumentOutOfRangeException(nameof(channels));

        int frames = interleavedSamples / channels;
        long ticks = checked((long)Math.Round(
            frames * TimestampFrequency / (double)sampleRate,
            MidpointRounding.AwayFromZero));
        Interlocked.Add(ref _timestamp, ticks);
        _utcNow = _utcNow.AddSeconds(frames / (double)sampleRate);
        FireDueTimers();
    }

    private bool ChangeTimer(
        BenchmarkTimer timer,
        TimeSpan dueTime,
        TimeSpan period)
    {
        long dueTicks = ToTimestampTicks(dueTime, nameof(dueTime));
        long periodTicks = period == TimeSpan.Zero
            ? long.MaxValue
            : ToTimestampTicks(period, nameof(period));
        lock (_timerSync)
        {
            if (timer.Disposed)
                return false;

            timer.DueTimestamp = dueTicks == long.MaxValue
                ? long.MaxValue
                : SaturatingAdd(_timestamp, dueTicks);
            timer.PeriodTicks = periodTicks;
            _timers.Add(timer);
            return true;
        }
    }

    private void DisposeTimer(BenchmarkTimer timer)
    {
        lock (_timerSync)
        {
            timer.Disposed = true;
            _timers.Remove(timer);
        }
    }

    private void FireDueTimers()
    {
        List<BenchmarkTimer>? due = null;
        lock (_timerSync)
        {
            long now = _timestamp;
            foreach (BenchmarkTimer timer in _timers.ToArray())
            {
                if (timer.Disposed || timer.DueTimestamp > now)
                    continue;

                due ??= [];
                due.Add(timer);
                if (timer.PeriodTicks == long.MaxValue)
                {
                    timer.DueTimestamp = long.MaxValue;
                    _timers.Remove(timer);
                }
                else
                {
                    timer.DueTimestamp = SaturatingAdd(now, timer.PeriodTicks);
                }
            }
        }

        if (due is null)
            return;
        foreach (BenchmarkTimer timer in due)
            timer.Invoke();
    }

    private long ToTimestampTicks(TimeSpan value, string parameterName)
    {
        if (value == Timeout.InfiniteTimeSpan)
            return long.MaxValue;
        if (value < TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(parameterName);

        double ticks = Math.Ceiling(
            value.TotalSeconds * TimestampFrequency);
        return ticks >= long.MaxValue ? long.MaxValue : (long)ticks;
    }

    private static long SaturatingAdd(long left, long right)
        => right >= long.MaxValue - left ? long.MaxValue : left + right;

    private sealed class BenchmarkTimer : ITimer
    {
        private readonly BenchmarkTimeProvider _owner;
        private readonly TimerCallback _callback;
        private readonly object? _state;

        public BenchmarkTimer(
            BenchmarkTimeProvider owner,
            TimerCallback callback,
            object? state)
        {
            _owner = owner;
            _callback = callback;
            _state = state;
        }

        public long DueTimestamp { get; set; } = long.MaxValue;
        public long PeriodTicks { get; set; } = long.MaxValue;
        public bool Disposed { get; set; }

        public bool Change(TimeSpan dueTime, TimeSpan period)
            => _owner.ChangeTimer(this, dueTime, period);

        public void Dispose() => _owner.DisposeTimer(this);

        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }

        public void Invoke()
        {
            lock (_owner._timerSync)
            {
                if (Disposed)
                    return;
            }
            _callback(_state);
        }
    }
}

internal sealed class BenchmarkClockedAecService :
    IAcousticEchoCancellationService,
    IAcousticProcessingMetricsSource
{
    private readonly IAcousticEchoCancellationService _inner;
    private readonly IAcousticProcessingMetricsSource _metricsSource;
    private readonly TimeProvider _timeProvider;

    public int DelaySamples => _inner.DelaySamples;

    public BenchmarkClockedAecService(
        IAcousticEchoCancellationService inner,
        IAcousticProcessingMetricsSource metricsSource,
        TimeProvider timeProvider)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _metricsSource = metricsSource ?? throw new ArgumentNullException(nameof(metricsSource));
        _timeProvider = timeProvider ?? throw new ArgumentNullException(nameof(timeProvider));
    }

    public void Process(
        Span<float> capture,
        ReadOnlySpan<float> render,
        uint channels,
        uint sampleRate)
        => _inner.Process(capture, render, channels, sampleRate);

    public void Reset() => _inner.Reset();

    public bool TryGetLatestMetrics(out AcousticProcessingMetrics metrics)
    {
        if (!_metricsSource.TryGetLatestMetrics(out metrics))
            return false;

        metrics = metrics with { TimestampTicks = _timeProvider.GetTimestamp() };
        return true;
    }

    public void Dispose()
    {
        // The evaluator owns the production AEC instance used by both this view and ingress.
    }
}

internal sealed class BenchmarkVadService : IVoiceActivityDetectService
{
    public event EventHandler<VADEventArgs>? OnSpeechEnded;
    public event EventHandler<EventArgs>? OnSpeechStarted;
    public event EventHandler<VADEventArgs>? OnSpeaking;
    public event EventHandler<VadProbabilityEventArgs>? OnProbability
    {
        add { }
        remove { }
    }

    public void Predict(ReadOnlySpan<float> audioData)
    {
    }

    public void EmitBootstrapUtterance()
    {
        OnSpeechStarted?.Invoke(this, EventArgs.Empty);
        OnSpeaking?.Invoke(
            this,
            new VADEventArgs([0.1f, 0.2f, 0.1f, 0.2f], VADEventDataType.Chunk));
        OnSpeechEnded?.Invoke(
            this,
            new VADEventArgs([], VADEventDataType.End, TimeSpan.FromMilliseconds(120)));
    }

    public void Reset()
    {
    }

    public void Dispose()
    {
    }
}

internal sealed class BenchmarkAsrService : IASRService
{
    private long _session;

    public Task<SessionInfo> StartSessionAsync(CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        string id = $"benchmark-{Interlocked.Increment(ref _session)}";
        return Task.FromResult(new SessionInfo(id));
    }

    public void FeedAudio(string sessionId, ReadOnlySpan<float> audioData)
    {
    }

    public Task<AsrResult?> FinishSessionAsync(string sessionId, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        return Task.FromResult<AsrResult?>(new AsrResult { Text = "开始测试" });
    }

    public Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        return Task.CompletedTask;
    }

    public void Dispose()
    {
    }
}

internal sealed class BenchmarkLlmService : IStreamLLMService
{
    public Task<LlmCompletionResult> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        return Task.FromResult(new LlmCompletionResult { Text = "正在播放。" });
    }

    public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.Yield();
        ct.ThrowIfCancellationRequested();
        yield return new LlmTextChunk
        {
            Text = "正在播放。",
            Kind = LlmChunkKind.Text,
            IsFinal = true,
        };
    }

    public void Dispose()
    {
    }
}

internal sealed class BenchmarkTtsService : IStreamTTSService
{
    private TaskCompletionSource _caseRelease =
        CompletedRelease();

    public int OutputSampleRate => EvaluationOptions.TargetSampleRate;
    public int OutputChannels => 1;

    public void BeginCase()
        => Interlocked.Exchange(
            ref _caseRelease,
            new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously));

    public void ReleaseCase()
        => Volatile.Read(ref _caseRelease).TrySetResult();

    public IReadOnlyList<TtsVoice> ListVoices() => [new TtsVoice { Name = "benchmark" }];

    public TtsVoice? GetDefaultVoice() => new TtsVoice { Name = "benchmark" };

    public Task<TtsResult> GetAudioAsync(
        string text,
        TtsSynthesisOptions? options = null,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        float[] samples = new float[320];
        return Task.FromResult(new TtsResult
        {
            Samples = samples,
            SampleRate = OutputSampleRate,
            Channels = OutputChannels,
            Voice = "benchmark",
        });
    }

    public async IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
        string text,
        TtsSynthesisOptions? options = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        Task release = Volatile.Read(ref _caseRelease).Task;
        await Task.Yield();
        ct.ThrowIfCancellationRequested();
        float[] samples = new float[320];
        yield return new TtsAudioChunk
        {
            Samples = samples,
            SampleRate = OutputSampleRate,
            Channels = OutputChannels,
            IsFinal = false,
            EmittedSeconds = samples.Length / (double)OutputSampleRate,
        };

        await release.WaitAsync(ct).ConfigureAwait(false);
    }

    public void Dispose()
    {
        ReleaseCase();
    }

    private static TaskCompletionSource CompletedRelease()
    {
        var completion = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        completion.SetResult();
        return completion;
    }
}
