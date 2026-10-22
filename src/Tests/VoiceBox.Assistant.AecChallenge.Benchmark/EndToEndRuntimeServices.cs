using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.LLMService.Models;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;

namespace VoiceBox.Assistant.AecChallenge.Benchmark;

internal sealed class BootstrapThenRealVadService : IVoiceActivityDetectService
{
    private readonly IVoiceActivityDetectService _inner;

    public BootstrapThenRealVadService(IVoiceActivityDetectService inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _inner.OnSpeechStarted += (_, args) => OnSpeechStarted?.Invoke(this, args);
        _inner.OnSpeaking += (_, args) => OnSpeaking?.Invoke(this, args);
        _inner.OnSpeechEnded += (_, args) => OnSpeechEnded?.Invoke(this, args);
        _inner.OnProbability += (_, args) => OnProbability?.Invoke(this, args);
    }

    public event EventHandler<VADEventArgs>? OnSpeechEnded;
    public event EventHandler<EventArgs>? OnSpeechStarted;
    public event EventHandler<VADEventArgs>? OnSpeaking;
    public event EventHandler<VadProbabilityEventArgs>? OnProbability;

    public void Predict(ReadOnlySpan<float> audioData)
        => _inner.Predict(audioData);

    public void Reset() => _inner.Reset();

    public void EmitBootstrapUtterance()
    {
        var audio = new float[EvaluationOptions.TargetSampleRate * 400 / 1000];
        Array.Fill(audio, 0.1f);
        OnSpeechStarted?.Invoke(this, EventArgs.Empty);
        OnSpeaking?.Invoke(
            this,
            new VADEventArgs(audio, VADEventDataType.Chunk));
        OnSpeechEnded?.Invoke(
            this,
            new VADEventArgs([], VADEventDataType.End, TimeSpan.FromMilliseconds(400)));
    }

    public void Dispose() => _inner.Dispose();
}

internal sealed class BootstrapThenRealAsrService : IASRService
{
    private readonly IASRService _inner;
    private readonly ConcurrentDictionary<string, byte> _bootstrapSessions =
        new(StringComparer.Ordinal);
    private int _nextSessionIsBootstrap;
    private long _bootstrapSession;

    public BootstrapThenRealAsrService(IASRService inner)
        => _inner = inner ?? throw new ArgumentNullException(nameof(inner));

    public void BeginCase()
        => Interlocked.Exchange(ref _nextSessionIsBootstrap, 1);

    public Task<SessionInfo> StartSessionAsync(CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        if (Interlocked.Exchange(ref _nextSessionIsBootstrap, 0) == 1)
        {
            string id = $"e2e-bootstrap-{Interlocked.Increment(ref _bootstrapSession)}";
            _bootstrapSessions.TryAdd(id, 0);
            return Task.FromResult(new SessionInfo(id));
        }
        return _inner.StartSessionAsync(ct);
    }

    public void FeedAudio(string sessionId, ReadOnlySpan<float> audioData)
    {
        if (!_bootstrapSessions.ContainsKey(sessionId))
            _inner.FeedAudio(sessionId, audioData);
    }

    public Task<AsrResult?> FinishSessionAsync(
        string sessionId,
        CancellationToken ct = default)
    {
        if (_bootstrapSessions.TryRemove(sessionId, out _))
        {
            ct.ThrowIfCancellationRequested();
            return Task.FromResult<AsrResult?>(
                new AsrResult { Text = "开始端到端测试" });
        }
        return _inner.FinishSessionAsync(sessionId, ct);
    }

    public Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
    {
        if (_bootstrapSessions.TryRemove(sessionId, out _))
        {
            ct.ThrowIfCancellationRequested();
            return Task.CompletedTask;
        }
        return _inner.CancelSessionAsync(sessionId, ct);
    }

    public void Dispose() => _inner.Dispose();
}

internal sealed class PreparedTtsService : IStreamTTSService
{
    private readonly int _outputSampleRate;
    private readonly int _outputChannels;
    private readonly int _chunkSamples;
    private TtsResult? _prepared;
    private TaskCompletionSource _release = CompletedRelease();

    public PreparedTtsService(int outputSampleRate, int outputChannels)
    {
        _outputSampleRate = outputSampleRate;
        _outputChannels = outputChannels;
        _chunkSamples = Math.Max(outputChannels, outputSampleRate / 10 * outputChannels);
    }

    public int OutputSampleRate => _outputSampleRate;
    public int OutputChannels => _outputChannels;

    public void BeginCase(TtsResult prepared)
    {
        if (prepared.SampleRate != OutputSampleRate ||
            prepared.Channels != OutputChannels)
        {
            throw new InvalidDataException(
                "The prepared TTS result format changed within one evaluator run.");
        }
        _prepared = prepared;
        Interlocked.Exchange(
            ref _release,
            new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously));
    }

    public void ReleaseCase()
        => Volatile.Read(ref _release).TrySetResult();

    public IReadOnlyList<TtsVoice> ListVoices()
        => [new TtsVoice { Name = _prepared?.Voice ?? "prepared-voxcpm" }];

    public TtsVoice? GetDefaultVoice()
        => new() { Name = _prepared?.Voice ?? "prepared-voxcpm" };

    public Task<TtsResult> GetAudioAsync(
        string text,
        TtsSynthesisOptions? options = null,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        return Task.FromResult(
            _prepared ?? throw new InvalidOperationException(
                "No VoxCPM TTS result was prepared for this case."));
    }

    public async IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(
        string text,
        TtsSynthesisOptions? options = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        TtsResult prepared = _prepared ?? throw new InvalidOperationException(
            "No VoxCPM TTS result was prepared for this case.");
        Task release = Volatile.Read(ref _release).Task;
        for (int offset = 0; offset < prepared.Samples.Length; offset += _chunkSamples)
        {
            ct.ThrowIfCancellationRequested();
            int length = Math.Min(_chunkSamples, prepared.Samples.Length - offset);
            var samples = new float[length];
            prepared.Samples.AsSpan(offset, length).CopyTo(samples);
            yield return new TtsAudioChunk
            {
                Samples = samples,
                SampleRate = prepared.SampleRate,
                Channels = prepared.Channels,
                EmittedSeconds =
                    (offset + length) /
                    (double)(prepared.SampleRate * prepared.Channels),
                IsFinal = false,
            };
            await Task.Yield();
        }
        await release.WaitAsync(ct).ConfigureAwait(false);
    }

    public void Dispose() => ReleaseCase();

    private static TaskCompletionSource CompletedRelease()
    {
        var result = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        result.SetResult();
        return result;
    }
}

internal sealed class EndToEndLlmService : IStreamLLMService
{
    private string _response = "正在播放端到端测试提示。";
    private int _generation;

    public void SetResponse(string response)
    {
        _response = string.IsNullOrWhiteSpace(response)
            ? throw new ArgumentException("TTS prompt cannot be empty.", nameof(response))
            : response;
        Interlocked.Exchange(ref _generation, 0);
    }

    public Task<LlmCompletionResult> CompleteAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        string text = Interlocked.Increment(ref _generation) == 1
            ? _response
            : string.Empty;
        return Task.FromResult(new LlmCompletionResult { Text = text });
    }

    public async IAsyncEnumerable<LlmTextChunk> StreamAsync(
        IReadOnlyList<ChatMessage> messages,
        LlmCompleteOptions? options = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.Yield();
        ct.ThrowIfCancellationRequested();
        if (Interlocked.Increment(ref _generation) != 1)
            yield break;
        yield return new LlmTextChunk
        {
            Text = _response,
            Kind = LlmChunkKind.Text,
            IsFinal = true,
        };
    }

    public void Dispose()
    {
    }
}
