using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.ASR.FireRedASR2.Internal;
using VoiceBox.ASR.FireRedASR2.Session;

namespace VoiceBox.ASR.FireRedASR2.Pump
{
    /// <summary>
    /// Drains a session's audio channel and turns each chunk into features via
    /// <see cref="OfflineStream.AddSamples(ReadOnlySpan{float})"/>. Runs as a
    /// long-running task; <see cref="StopAsync"/> completes the channel and awaits drain.
    /// </summary>
    internal sealed class FireRedFeaturePump
    {
        private readonly FireRedSession _session;
        private readonly ILogger _logger;
        private Task? _runTask;
        private int _started;

        public FireRedFeaturePump(FireRedSession session, ILogger? logger = null)
        {
            ArgumentNullException.ThrowIfNull(session);
            _session = session;
            _logger = logger ?? NullLogger.Instance;
        }

        public Task RunTask => _runTask ?? Task.CompletedTask;

        public void Start()
        {
            if (Interlocked.Exchange(ref _started, 1) != 0)
                throw new InvalidOperationException("Pump already started.");

            _runTask = Task.Run(() => RunAsync(_session.WorkCancellationToken));
        }

        public async Task StopAsync(CancellationToken ct = default)
        {
            _session.CompleteWriter();
            if (_runTask is null) return;

            try
            {
                await _runTask.WaitAsync(ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Drain timed out (or caller cancelled). Force the pump out of its read loop
                // and wait — unbounded — for it to actually exit. Skipping this step leaves
                // the pump task running concurrently with the synchronous recognition pass
                // and the subsequent OfflineStream.Dispose, causing AddSamples-vs-Dispose
                // races and silent ObjectDisposedException churn.
                _session.AbortWork();
                try
                {
                    await _runTask.ConfigureAwait(false);
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    try { _logger.LogWarning(ex, "ASR feature pump faulted during forced stop (session: {SessionId}).", _session.SessionId); }
                    catch { /* Preserve the drain cancellation below. */ }
                }

                // Preserve the timeout/caller cancellation after the pump has been
                // synchronised. Otherwise FinishSessionAsync proceeds with a partial
                // stream and returns a result even though draining was cancelled.
                throw;
            }
            catch (Exception ex)
            {
                try { _logger.LogWarning(ex, "ASR feature pump faulted on stop (session: {SessionId}).", _session.SessionId); }
                catch { /* Preserve the feature failure below. */ }
                throw;
            }
        }

        private async Task RunAsync(CancellationToken ct)
        {
            try
            {
                while (await _session.WaitToReadAsync(ct).ConfigureAwait(false))
                {
                    // Inner loop must also honour cancellation: when StopAsync's timeout
                    // path forces Cancel() while the channel still holds dozens of pooled
                    // chunks, draining them all (each ~tens-of-ms of FBank work) delays
                    // pump exit and the subsequent stream Dispose. The pooled items are
                    // still disposed in finally so no buffer is leaked.
                    while (!ct.IsCancellationRequested && _session.TryReadAudio(out var pooled))
                    {
                        try { ConsumeAudio(pooled); }
                        finally { pooled.Dispose(); }
                    }
                    if (ct.IsCancellationRequested) break;
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { }
            catch (ChannelClosedException) { }
            catch (Exception ex)
            {
                _ = _session.FailAsync(ex);
                try { _logger.LogError(ex, "ASR feature pump faulted (session: {SessionId}).", _session.SessionId); }
                catch { /* Preserve the feature failure even if logging also fails. */ }
            }
            finally
            {
                // Drain any unconsumed pooled audio so its rented buffers go back to the
                // pool even when we exit early due to cancellation.
                while (_session.TryReadAudio(out var leftover))
                    leftover.Dispose();
            }
        }

        private void ConsumeAudio(PooledAudio pooled)
        {
            ReadOnlySpan<float> span = pooled.Span;
            if (span.IsEmpty) return;

            // A failed feature append loses this chunk; the final pass cannot recover
            // it from raw audio. Let the outer loop terminate the failed session.
            _session.Stream.AddSamples(span);
        }
    }
}
