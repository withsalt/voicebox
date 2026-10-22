using Microsoft.Extensions.Logging;
using System.Runtime.CompilerServices;
using VoiceBox.Assistant.Api.Playback;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Contracts;
using VoiceBox.Assistant.Modules.BargeIn;
using VoiceBox.Assistant.Modules.Response.Contracts;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Assistant.Realtime.Resampling;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Playback.Adapters;
using VoiceBox.Assistant.Modules.Playback.Models;

namespace VoiceBox.Assistant.Modules.Playback.Controllers;

/// <summary>
/// Owns the complete synthesis/playback data-plane operation: sentence segmentation, TTS stream,
/// bounded queue backpressure, preroll, drain and the per-turn resampler lease. ResponseModule
/// supplies speakable text; it does not create or dispose playback sessions directly.
/// </summary>
internal sealed class PlaybackModule :
    IResponsePlaybackSessionFactory,
    ISpeechPlaybackControlPort,
    IPlaybackCancellationPort,
    ISpeechPlaybackRequestExecutor,
    IDisposable
{
    private readonly ISpeechSynthesisStreamPort _speechSynthesis;
    private readonly IAssistantAudioOutput _playback;
    private readonly ILogger _logger;
    private readonly VoiceAssistantOptions _options;
    private readonly AudioResamplerPool _resamplers;
    private readonly IPlaybackStartedSink _playbackStarted;
    private readonly IScheduledSpeechSink _scheduledSpeech;
    private readonly BoundedPlaybackPortAdapter _devicePlayback;
    private readonly SpeechPlaybackCoordinator _coordinator;
    private int _disposed;

    public PlaybackModule(
        ISpeechSynthesisStreamPort speechSynthesis,
        IAssistantAudioOutput playback,
        ILogger logger,
        VoiceAssistantOptions options,
        AudioResamplerPool resamplers,
        IPlaybackStartedSink playbackStarted,
        IScheduledSpeechSink scheduledSpeech)
    {
        _speechSynthesis = speechSynthesis ?? throw new ArgumentNullException(nameof(speechSynthesis));
        _playback = playback ?? throw new ArgumentNullException(nameof(playback));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _resamplers = resamplers ?? throw new ArgumentNullException(nameof(resamplers));
        _playbackStarted = playbackStarted ?? throw new ArgumentNullException(nameof(playbackStarted));
        _scheduledSpeech = scheduledSpeech ?? throw new ArgumentNullException(nameof(scheduledSpeech));
        _devicePlayback = new BoundedPlaybackPortAdapter(_playback);
        _coordinator = new SpeechPlaybackCoordinator(this, _devicePlayback);
    }

    public IResponsePlaybackSession Create(long turnId)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (turnId <= 0)
            throw new ArgumentOutOfRangeException(nameof(turnId), "Turn id must be positive.");
        return new QueuedResponsePlaybackSession(this, turnId);
    }

    /// <summary>
    /// Starts the single logical playback worker. The returned task represents the worker lifetime
    /// and is registered with <see cref="Runtime.Lifecycle.Controllers.AssistantTaskSupervisor"/> by the shell.
    /// </summary>
    public Task Start(CancellationToken runCancellation)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _coordinator.Start(runCancellation);
    }

    public Task StopAsync() => _coordinator.StopAsync();

    public async Task<string> PlayAsync(
        long turnId,
        IAsyncEnumerable<SpeakableTextDelta> text,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (turnId <= 0)
            throw new ArgumentOutOfRangeException(nameof(turnId), "Turn id must be positive.");
        ArgumentNullException.ThrowIfNull(text);

        SpeechEnqueueResult submission = await _coordinator.EnqueueAsync(
            turnId,
            text,
            cancellationToken).ConfigureAwait(false);
        return await AwaitTextAsync(submission).ConfigureAwait(false);
    }

    async ValueTask<SpeechEnqueueResult> ISpeechPlaybackControlPort.EnqueueSpeechAsync(
        string text,
        SpeechPlaybackQueuePosition queuePosition,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(text);
        cancellationToken.ThrowIfCancellationRequested();

        return await _coordinator.EnqueueAsync(
            ownerTurnId: null,
            ToTextStream(text, cancellationToken),
            cancellationToken,
            queuePosition).ConfigureAwait(false);
    }

    void ISpeechPlaybackControlPort.StopCurrentSpeechPlayback()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _coordinator.TryStopActivePlayback();
    }

    void IPlaybackCancellationPort.CancelActivePlayback()
        => _coordinator.CancelActivePlayback();

    PlaybackStopOutcome IPlaybackCancellationPort.TryStopActivePlayback(long expectedOwnerTurnId)
        => _coordinator.TryStopActivePlayback(expectedOwnerTurnId);

    bool IPlaybackCancellationPort.TryDuckActivePlayback(long expectedOwnerTurnId, float gain, float originalGain)
        => _coordinator.TryDuckActivePlayback(expectedOwnerTurnId, gain, originalGain);

    bool IPlaybackCancellationPort.TryRestoreActivePlayback(long expectedOwnerTurnId)
        => _coordinator.TryRestoreActivePlayback(expectedOwnerTurnId);

    long? IPlaybackCancellationPort.ActiveOwnerTurnId => _coordinator.ActiveOwnerTurnId;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _coordinator.Dispose();
    }

    async Task<string> ISpeechPlaybackRequestExecutor.ExecuteAsync(
        SpeechPlaybackRequest request,
        CancellationToken cancellationToken)
    {
        AudioResamplerPool.Lease lease = _resamplers.Acquire();
        try
        {
            long? turnId = request.OwnerTurnId;
            var pump = new TtsPlaybackPump(
                _speechSynthesis,
                _devicePlayback,
                _logger,
                _options.MinSentenceChars,
                _options.PlaybackTailBufferMs,
                _options.PlaybackPrerollMs,
                lease.Resampler,
                turnId,
                _playbackStarted,
                _scheduledSpeech);
            return await pump.RunAsync(request.Text, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            lease.Dispose();
        }
    }

    private static async Task<string> AwaitTextAsync(SpeechEnqueueResult submission)
    {
        Task<string> completion = (Task<string>)submission.Completion;
        return await completion.ConfigureAwait(false);
    }

    private static async IAsyncEnumerable<SpeakableTextDelta> ToTextStream(
        string text,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!string.IsNullOrEmpty(text))
            yield return new SpeakableTextDelta(text);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private sealed class QueuedResponsePlaybackSession(
        PlaybackModule owner,
        long turnId) : IResponsePlaybackSession
    {
        private readonly PlaybackModule _owner = owner;
        private readonly long _turnId = turnId;
        private int _disposed;

        public Task<string> PlayAsync(
            IAsyncEnumerable<SpeakableTextDelta> text,
            CancellationToken cancellationToken)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _owner.PlayAsync(_turnId, text, cancellationToken);
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _disposed, 1);
        }
    }
}
