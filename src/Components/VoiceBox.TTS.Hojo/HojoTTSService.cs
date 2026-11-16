using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.TTSService.Models;
using VoiceBox.TTS.Hojo.Internal;

namespace VoiceBox.TTS.Hojo;

/// <summary>Native Hojo 80M / 40M V2 inference, 24kHz mono. Bounded request and PCM queues serialize inference.</summary>
public sealed class HojoTTSService : IStreamTTSService, IAsyncDisposable
{
    private readonly HojoTTSOptions _options;
    private readonly ILogger<HojoTTSService>? _logger;
    private readonly Func<IHojoPipeline> _factory;
    private readonly object _lifetime = new(), _initialization = new();
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly CancellationTokenSource _shutdown = new();
    private readonly TaskCompletionSource _drained = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private IHojoPipeline? _pipeline;
    private Task? _disposeTask;
    private int _active;

    public HojoTTSService(IOptions<HojoTTSOptions> options, ILogger<HojoTTSService>? logger = null)
        : this(options?.Value ?? throw new ArgumentNullException(nameof(options)), null, logger) { }

    internal HojoTTSService(HojoTTSOptions options, Func<IHojoPipeline>? factory, ILogger<HojoTTSService>? logger = null)
    {
        _options = options.Snapshot();
        _factory = factory ?? (() => new HojoPipeline(_options));
        _logger = logger;
    }

    public int OutputSampleRate => 24000;
    public int OutputChannels => 1;

    public IReadOnlyList<TtsVoice> ListVoices()
    {
        using var request = Admit(CancellationToken.None);
        try { return EnsurePipeline().Voices; }
        finally { ReleaseRequest(); }
    }

    public TtsVoice? GetDefaultVoice() => ListVoices().FirstOrDefault(v => v.Name == _options.DefaultVoice);

    private IHojoPipeline EnsurePipeline()
    {
        lock (_initialization)
        {
            if (_pipeline is not null) return _pipeline;
            var watch = Stopwatch.StartNew();
            _pipeline = _factory();
            _logger?.LogInformation("Hojo model loaded in {Milliseconds} ms", watch.ElapsedMilliseconds);
            return _pipeline;
        }
    }

    public IAsyncEnumerable<TtsAudioChunk> GetStreamAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length > _options.MaxTextLength) throw new ArgumentException("Text is too long; split it into shorter utterances.", nameof(text));
        var request = HojoRequest.Create(_options, options);
        return StreamAsync(text, request, ct);
    }

    public async Task<TtsResult> GetAudioAsync(string text, TtsSynthesisOptions? options = null, CancellationToken ct = default)
    {
        // Snapshot before awaiting, so caller mutations cannot change an admitted request.
        var request = HojoRequest.Create(_options, options);
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length > _options.MaxTextLength) throw new ArgumentException("Text is too long.", nameof(text));
        var watch = Stopwatch.StartNew();
        var chunks = new List<float[]>();
        int length = 0;
        await foreach (var chunk in StreamAsync(text, request, ct).ConfigureAwait(false))
        {
            if (chunk.Samples.Length == 0) continue;
            length = checked(length + chunk.Samples.Length);
            chunks.Add(chunk.Samples);
        }
        var samples = new float[length];
        int offset = 0;
        foreach (float[] chunk in chunks) { chunk.CopyTo(samples, offset); offset += chunk.Length; }
        return new TtsResult { Samples = samples, SampleRate = 24000, Channels = 1, Voice = request.Voice, ElapsedSeconds = watch.Elapsed.TotalSeconds };
    }

    private async IAsyncEnumerable<TtsAudioChunk> StreamAsync(string text, HojoRequest request, [EnumeratorCancellation] CancellationToken ct)
    {
        using var linked = Admit(ct);
        var channel = Channel.CreateBounded<TtsAudioChunk>(new BoundedChannelOptions(_options.BufferedChunks)
        {
            SingleReader = true, SingleWriter = true, FullMode = BoundedChannelFullMode.Wait,
            AllowSynchronousContinuations = false,
        });
        // Always start the producer, even when cancellation races Task.Run; its finally owns admission release.
        Task producer = Task.Run(() => ProduceAsync(text, request, channel.Writer, linked.Token));
        try
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(linked.Token).ConfigureAwait(false))
            {
                linked.Token.ThrowIfCancellationRequested();
                yield return chunk;
            }
        }
        finally
        {
            await linked.CancelAsync().ConfigureAwait(false);
            await producer.ConfigureAwait(false);
        }
    }

    private async Task ProduceAsync(string text, HojoRequest request, ChannelWriter<TtsAudioChunk> writer, CancellationToken ct)
    {
        bool entered = false;
        Exception? failure = null;
        try
        {
            await _gate.WaitAsync(ct).ConfigureAwait(false);
            entered = true;
            var pipeline = EnsurePipeline();
            ct.ThrowIfCancellationRequested();
            long emitted = 0;
            foreach (float[] samples in pipeline.Generate(text, request, ct))
            {
                ct.ThrowIfCancellationRequested();
                if (samples.Length == 0) continue;
                emitted += samples.Length;
                await writer.WriteAsync(new TtsAudioChunk
                {
                    Samples = samples, SampleRate = 24000, Channels = 1, EmittedSeconds = (double)emitted / 24000,
                }, ct).ConfigureAwait(false);
            }
            ct.ThrowIfCancellationRequested();
            await writer.WriteAsync(new TtsAudioChunk
            {
                Samples = [], SampleRate = 24000, Channels = 1, EmittedSeconds = (double)emitted / 24000, IsFinal = true,
            }, ct).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            failure = ct.IsCancellationRequested && ex is OperationCanceledException or OnnxRuntimeException
                ? new OperationCanceledException(ct) : ex;
            if (failure is not OperationCanceledException) _logger?.LogError(failure, "Hojo synthesis failed");
        }
        finally
        {
            if (entered) _gate.Release();
            writer.TryComplete(failure);
            ReleaseRequest();
        }
    }

    private CancellationTokenSource Admit(CancellationToken ct)
    {
        lock (_lifetime)
        {
            ObjectDisposedException.ThrowIf(_disposeTask is not null, this);
            ct.ThrowIfCancellationRequested();
            if (_active >= _options.MaxPendingRequests) throw new InvalidOperationException("Hojo request capacity exceeded; cancel obsolete speech or retry later.");
            var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _shutdown.Token);
            _active++;
            return linked;
        }
    }

    private void ReleaseRequest()
    {
        lock (_lifetime)
        {
            _active--;
            if (_active == 0 && _disposeTask is not null) _drained.TrySetResult();
        }
    }

    public ValueTask DisposeAsync()
    {
        lock (_lifetime)
        {
            if (_disposeTask is null)
            {
                _disposeTask = Task.Run(DisposeCoreAsync);
                if (_active == 0) _drained.TrySetResult();
            }
            return new ValueTask(_disposeTask);
        }
    }

    private async Task DisposeCoreAsync()
    {
        await _shutdown.CancelAsync().ConfigureAwait(false);
        await _drained.Task.ConfigureAwait(false);
        // No timeout-based forced disposal: native inference must exit before sessions can be freed.
        try { _pipeline?.Dispose(); }
        finally { _gate.Dispose(); _shutdown.Dispose(); }
    }

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();
}
