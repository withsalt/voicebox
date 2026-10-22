using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using VoiceBox.ASR.Confucius4.Internal;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.Confucius4.Streaming;

internal sealed class Confucius4Session : AsrSession
{
    private readonly object _sync = new();
    private readonly Confucius4AsrOptions _options;
    private readonly IConfucius4Runtime _runtime;
    private readonly SemaphoreSlim _inference;
    // Audio callbacks have no async backpressure path. Keep the sample budget as
    // the limit, but do not impose a second fixed callback-count limit: WASAPI and
    // PortAudio callback sizes vary widely between devices.
    private readonly Channel<AudioBlock> _input = Channel.CreateUnbounded<AudioBlock>(
        new UnboundedChannelOptions { SingleReader = true, AllowSynchronousContinuations = false });
    // Results are cumulative snapshots: replacing an older snapshot cannot lose committed text.
    private readonly Channel<AsrResult> _output = Channel.CreateBounded<AsrResult>(new BoundedChannelOptions(32)
        { SingleWriter = true, FullMode = BoundedChannelFullMode.DropOldest, AllowSynchronousContinuations = false });
    private readonly float[] _audio;
    private readonly IncrementalMelState _mel = new();
    private int _queuedSamples;
    private int _samples;
    private int _nextDecode;
    private int _readerClaimed;
    private int _enumeratorClaimed;
    private bool _accepting = true;
    private string _prefix = "";
    private string _stable = "";
    private string _archive = "";
    private string _language = "";
    private AsrResult _latest = Transcript.Result("", "", "");

    public Task<AsrResult> Completion { get; private set; } =
        Task.FromResult(Transcript.Result("", "", ""));

    public Confucius4Session(string sessionId, Confucius4AsrOptions options, IConfucius4Runtime runtime, SemaphoreSlim inference)
        : base(sessionId)
    {
        _options = options;
        _runtime = runtime;
        _inference = inference;
        _audio = new float[checked(options.MaxWindowMilliseconds * 16)];
        _nextDecode = (options.ChunkMilliseconds + options.LookaheadMilliseconds) * 16;
    }

    protected override Task StartSessionAsync(CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        Completion = Task.Run(RunAsync);
        return Task.CompletedTask;
    }

    public override void FeedAudio(ReadOnlySpan<float> samples)
    {
        lock (_sync)
        {
            if (!_accepting || FailureException is not null)
                throw new InvalidOperationException("The session is finishing or has stopped.");
            if (samples.IsEmpty) return;
            if (samples.Length > _options.MaxBufferedMilliseconds * 16 - Volatile.Read(ref _queuedSamples))
                throw new InvalidOperationException("Confucius4 audio queue is full. Slow the producer or provision more inference capacity.");
            foreach (float sample in samples)
                if (!float.IsFinite(sample)) throw new ArgumentException("PCM must contain only finite samples.");
            float[] buffer = ArrayPool<float>.Shared.Rent(samples.Length);
            samples.CopyTo(buffer);
            Interlocked.Add(ref _queuedSamples, samples.Length);
            if (!_input.Writer.TryWrite(new(buffer, samples.Length)))
            {
                Interlocked.Add(ref _queuedSamples, -samples.Length);
                ArrayPool<float>.Shared.Return(buffer);
                throw new InvalidOperationException("Confucius4 audio queue is full or closed.");
            }
        }
    }

    private void Complete()
    {
        lock (_sync)
        {
            _accepting = false;
            _input.Writer.TryComplete();
        }
    }

    protected override bool RetainAfterFinishWaitCancellation => true;

    protected override async Task<AsrResult?> FinishSessionAsync()
    {
        Complete();
        return await Completion.ConfigureAwait(false);
    }

    protected override Task CancelSessionAsync()
    {
        Complete();
        return Task.CompletedTask;
    }

    protected override void OnSessionFaulted(Exception error)
    {
        Complete();
        _output.Writer.TryComplete(error);
    }

    protected override async Task ReleaseSessionAsync()
    {
        try { await Completion.ConfigureAwait(false); }
        catch { }
    }

    public IAsyncEnumerable<AsrResult> Read(CancellationToken token)
    {
        if (Interlocked.Exchange(ref _readerClaimed, 1) != 0)
            throw new InvalidOperationException("Each session supports one result consumer.");
        return ReadOnce(token);
    }

    private async IAsyncEnumerable<AsrResult> ReadOnce([EnumeratorCancellation] CancellationToken token)
    {
        if (Interlocked.Exchange(ref _enumeratorClaimed, 1) != 0)
            throw new InvalidOperationException("Each session supports one result consumer.");
        await foreach (var result in _output.Reader.ReadAllAsync(token).ConfigureAwait(false))
            yield return result;
    }

    private async Task<AsrResult> RunAsync()
    {
        Exception? failure = null;
        try
        {
            while (await _input.Reader.WaitToReadAsync(CancellationToken).ConfigureAwait(false))
            {
                while (_input.Reader.TryRead(out AudioBlock block))
                {
                    Interlocked.Add(ref _queuedSamples, -block.Length);
                    try
                    {
                        bool decodeDue = await AppendAudioAsync(block.Buffer, block.Length, false).ConfigureAwait(false);
                        if (decodeDue && _samples > 0)
                            await DecodeAsync(final: false).ConfigureAwait(false);
                    }
                    finally { ArrayPool<float>.Shared.Return(block.Buffer); }
                }
            }
            // Finalize even when the input ended exactly at a chunk boundary.
            if (_samples > 0) await DecodeAsync(final: true).ConfigureAwait(false);
            return _latest;
        }
        catch (Exception ex)
        {
            failure = ex;
            if (ex is not OperationCanceledException || !CancellationToken.IsCancellationRequested)
                _ = FailAsync(ex);
            throw;
        }
        finally
        {
            Complete();
            while (_input.Reader.TryRead(out AudioBlock block)) ArrayPool<float>.Shared.Return(block.Buffer);
            _output.Writer.TryComplete(failure);
        }
    }

    private async Task<bool> AppendAudioAsync(float[] samples, int length, bool decodeDue)
    {
        int offset = 0;
        int chunkSamples = _options.ChunkMilliseconds * 16;
        while (offset < length)
        {
            CancellationToken.ThrowIfCancellationRequested();
            int target = Math.Min(_nextDecode, _audio.Length);
            int count = Math.Min(target - _samples, length - offset);
            samples.AsSpan(offset, count).CopyTo(_audio.AsSpan(_samples));
            _samples += count;
            offset += count;

            if (_samples == _audio.Length)
            {
                await DecodeAsync(final: true).ConfigureAwait(false);
                _archive = _latest.Text;
                _samples = 0;
                _prefix = _stable = "";
                _mel.Reset();
                _nextDecode = (_options.ChunkMilliseconds + _options.LookaheadMilliseconds) * 16;
                decodeDue = false;
            }
            else if (_samples == _nextDecode)
            {
                decodeDue = true;
                // The next available decode is relative to the newest audio that
                // will be consumed in this batch, so multiple crossed boundaries
                // collapse into one catch-up inference.
                _nextDecode = Math.Min(_audio.Length, _nextDecode + chunkSamples);
            }
        }
        return decodeDue;
    }

    private async Task DecodeAsync(bool final)
    {
        await _inference.WaitAsync(CancellationToken).ConfigureAwait(false);
        try
        {
            string raw = Transcript.Normalize(_runtime.Decode(_audio.AsSpan(0, _samples), _prefix,
                final ? _options.FinalMaxNewTokens : _options.MaxNewTokens, _mel, CancellationToken, final));
            CancellationToken.ThrowIfCancellationRequested();
            string proposed = final ? raw : _runtime.Rollback(raw, _options.RollbackTokens);
            // BPE re-tokenization can move a token boundary backwards; never retract a submitted prefix.
            if (proposed.StartsWith(_prefix, StringComparison.Ordinal)) _prefix = proposed;
            var (language, text) = Transcript.Parse(raw, _options.Language);
            var (_, stable) = Transcript.Parse(_prefix, _options.Language);
            if (stable.StartsWith(_stable, StringComparison.Ordinal)) _stable = stable;
            if (!text.StartsWith(_stable, StringComparison.Ordinal)) text = _stable;
            if (language.Length > 0) _language = language;
            string committed = Transcript.Join(_archive, _stable);
            string all = Transcript.Join(_archive, text);
            if (final) committed = all;
            _latest = Transcript.Result(_language, all, committed);
            // AsrResult is mutable in VoiceBox.Core; keep consumer mutations out of session state.
            lock (_sync)
            {
                CancellationToken.ThrowIfCancellationRequested();
                _output.Writer.TryWrite(Transcript.Result(_language, all, committed));
            }
        }
        finally { _inference.Release(); }
    }

    private readonly record struct AudioBlock(float[] Buffer, int Length);
}
