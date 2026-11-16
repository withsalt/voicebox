using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using VoiceBox.ASR.Confucius4.Internal;
using VoiceBox.ASR.Confucius4.Streaming;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.Confucius4;

/// <summary>Local ONNX ASR. FeedAudio copies input; one inference gate protects shared native state.</summary>
public sealed class Confucius4ASRService : BaseASRService, IStreamASRService
{
    private readonly object _offlineSync = new();
    private readonly Confucius4AsrOptions _options;
    private readonly IConfucius4Runtime _runtime;
    private readonly SemaphoreSlim _inference = new(1, 1);
    private readonly HashSet<Task> _offline = new();
    private readonly CancellationTokenSource _stop = new();

    public Confucius4ASRService(IOptions<Confucius4AsrOptions> options, ILogger<Confucius4ASRService> logger)
        : this(options.Value, logger, null) { }

    internal Confucius4ASRService(Confucius4AsrOptions options, ILogger logger, IConfucius4Runtime? runtime)
        : base(GetMaxConcurrentSessions(options))
    {
        _options = options.Snapshot();
        _runtime = runtime ?? new Confucius4Runtime(_options, logger);
        Warmup();
    }

    protected override AsrSession CreateSession(string sessionId)
        => new Confucius4Session(sessionId, _options, _runtime, _inference);

    public IAsyncEnumerable<AsrResult> GetStreamResultAsync(string sessionId, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
        EnsureServiceActive();
        if (!TryGetSession<Confucius4Session>(sessionId, out var session))
            throw new InvalidOperationException($"Session not found: {sessionId}");
        return session.Read(ct);
    }

    /// <summary>One-shot recognition of at most MaxWindowMilliseconds. Input is copied before returning.</summary>
    public Task<AsrResult> TranscribeAsync(ReadOnlyMemory<float> audio, CancellationToken ct = default)
    {
        lock (_offlineSync)
        {
            EnsureServiceActive();
            ct.ThrowIfCancellationRequested();
            // A caller can resume before the completion continuation removes its task.
            // Only unfinished recognitions occupy pending capacity.
            _offline.RemoveWhere(task => task.IsCompleted);
            if (_offline.Count >= _options.MaxConcurrentSessions)
                throw new InvalidOperationException("Maximum pending Confucius4 recognition count reached.");
            if (audio.Length > _options.MaxWindowMilliseconds * 16)
                throw new ArgumentException("Audio exceeds the configured window; use streaming sessions.");
            foreach (float sample in audio.Span)
                if (!float.IsFinite(sample)) throw new ArgumentException("PCM must contain only finite samples.");
            float[] copy = audio.ToArray();
            Task<AsrResult> task = Task.Run(async () =>
            {
                using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _stop.Token);
                await _inference.WaitAsync(linked.Token).ConfigureAwait(false);
                try
                {
                    string raw = copy.Length == 0 ? "" : Transcript.Normalize(_runtime.Decode(copy, "",
                        _options.FinalMaxNewTokens, new IncrementalMelState(), linked.Token, final: true));
                    var (language, text) = Transcript.Parse(raw, _options.Language);
                    return Transcript.Result(language, text, text);
                }
                finally { _inference.Release(); }
            });
            _offline.Add(task);
            _ = task.ContinueWith(completed => { lock (_offlineSync) _offline.Remove(completed); },
                CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.Default);
            return task;
        }
    }

    protected override void OnDisposeRequested()
    {
        _stop.Cancel();
    }

    private void Warmup()
    {
        try
        {
            _runtime.Decode(new float[5120], "", 1, new IncrementalMelState(), CancellationToken.None);
        }
        catch
        {
            _runtime.Dispose();
            throw;
        }
    }

    protected override void DisposeProvider()
    {
        Task[] offline;
        lock (_offlineSync)
        {
            offline = _offline.ToArray();
        }
        try { Task.WhenAll(offline).GetAwaiter().GetResult(); }
        catch { /* Individual session/recognition tasks retain their errors for their callers. */ }
        try { _runtime.Dispose(); }
        finally { _inference.Dispose(); _stop.Dispose(); }
    }

    private static int GetMaxConcurrentSessions(Confucius4AsrOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return options.MaxConcurrentSessions;
    }
}
