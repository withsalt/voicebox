using System;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.ASR.Qwen3.Internal;
using VoiceBox.ASR.Qwen3.Session;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.Qwen3.Pump
{
    /// <summary>
    /// Per-session background task that:
    ///   1. consumes audio chunks from the session's ingress channel,
    ///   2. appends them to the accumulated waveform buffer,
    ///   3. whenever the accumulated length crosses a chunk boundary,
    ///      requests a re-transcription via the shared <see cref="Qwen3AsrRuntime"/>,
    ///   4. publishes the resulting <see cref="AsrResult"/> on the session's result
    ///      channel for the <c>GetStreamResultAsync</c> consumer.
    ///
    /// Concurrency between sessions is serialised by a <see cref="SemaphoreSlim"/>
    /// supplied by the service (the recognition gate) — Qwen3-ASR runs heavy ONNX
    /// graphs and naive parallelism would only thrash the CPU/GPU.
    /// </summary>
    internal sealed class Qwen3AsrAudioPump
    {
        private readonly Qwen3AsrSession _session;
        private readonly IQwen3AsrRuntime _runtime;
        private readonly SemaphoreSlim _recognitionGate;
        private readonly ILogger _logger;
        private readonly System.Diagnostics.Stopwatch _sessionSw = System.Diagnostics.Stopwatch.StartNew();
        private readonly IncrementalMelState _melState = new();
        private Task? _runTask;
        private int _started;
        private string _accumulatedRaw = string.Empty;

        // Aggregated metrics for the session-level summary.
        private int _transcribeCount;
        private double _totalAudioSeconds;
        private double _totalInferenceSeconds;
        private double _totalEncodeSeconds;
        private double _totalPrefillSeconds;
        private double _totalDecodeSeconds;
        private bool _firstResultPublished;
        private double _firstResultElapsedSeconds;
        private double _firstResultAudioSeconds;

        public Qwen3AsrAudioPump(
            Qwen3AsrSession session,
            IQwen3AsrRuntime runtime,
            SemaphoreSlim recognitionGate,
            ILogger? logger = null)
        {
            ArgumentNullException.ThrowIfNull(session);
            ArgumentNullException.ThrowIfNull(runtime);
            ArgumentNullException.ThrowIfNull(recognitionGate);
            _session = session;
            _runtime = runtime;
            _recognitionGate = recognitionGate;
            _logger = logger ?? NullLogger.Instance;
        }

        public Task RunTask => _runTask ?? Task.CompletedTask;

        internal void ResetState()
        {
            _melState.Reset();
            _accumulatedRaw = string.Empty;
        }

        /// <summary>Aggregated metrics for the session, used by the service to log a finish summary.</summary>
        public Qwen3AsrSessionMetrics GetMetricsSnapshot() => new Qwen3AsrSessionMetrics(
            ElapsedSeconds: _sessionSw.Elapsed.TotalSeconds,
            TranscribeCount: _transcribeCount,
            TotalAudioSeconds: _totalAudioSeconds,
            TotalInferenceSeconds: _totalInferenceSeconds,
            TotalEncodeSeconds: _totalEncodeSeconds,
            TotalPrefillSeconds: _totalPrefillSeconds,
            TotalDecodeStepSeconds: _totalDecodeSeconds,
            FirstResultPublished: _firstResultPublished,
            FirstResultElapsedSeconds: _firstResultElapsedSeconds,
            FirstResultAudioSeconds: _firstResultAudioSeconds);

        public void Start()
        {
            if (Interlocked.Exchange(ref _started, 1) != 0)
                throw new InvalidOperationException("Pump already started.");
            _runTask = Task.Run(() => RunAsync(_session.WorkCancellationToken));
        }

        public async Task StopAsync(CancellationToken ct)
        {
            _session.MarkFinishing();
            if (_runTask is null) return;
            try { await _runTask.WaitAsync(ct).ConfigureAwait(false); }
            catch (OperationCanceledException)
            {
                _session.AbortWork();
                // The final pass and teardown must not race the pump's mel state/native work.
                await _runTask.ConfigureAwait(false);
                throw;
            }
            catch (Exception ex)
            {
                try { _logger.LogWarning(ex, "Qwen3 ASR pump faulted during stop (session: {SessionId}).", _session.SessionId); }
                catch { /* Preserve the pump failure below. */ }
                throw;
            }
        }

        /// <summary>
        /// Trigger one synchronous final transcription pass over whatever audio is
        /// currently accumulated. Used by <c>FinishSessionAsync</c> after the pump
        /// has been drained.
        /// </summary>
        public AsrResult RunFinalTranscription(CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            float[] waveform = _session.SnapshotAccumulated();
            if (waveform.Length == 0)
                return new AsrResult { Text = _session.LatestText, Language = _session.LatestLanguage };

            return TranscribeAndPublish(waveform, publishToStream: false, ct);
        }

        private async Task RunAsync(CancellationToken ct)
        {
            try
            {
                while (await _session.WaitToReadAsync(ct).ConfigureAwait(false))
                {
                    while (!ct.IsCancellationRequested && _session.TryReadAudio(out var pooled))
                    {
                        try { _session.AppendSamples(pooled.Span); }
                        finally { pooled.Dispose(); }
                    }
                    MaybeTranscribe(ct);
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { }
            catch (ChannelClosedException) { }
            catch (Exception ex)
            {
                _ = _session.FailAsync(ex);
                try { _logger.LogError(ex, "Qwen3 ASR pump faulted (session: {SessionId}).", _session.SessionId); }
                catch { /* Logging must not replace the original error or prevent cleanup. */ }
            }
            finally
            {
                _session.DrainAudio();
            }
        }

        private void MaybeTranscribe(CancellationToken ct)
        {
            int requiredLen = (_session.ChunkId + 1) * _session.ChunkSizeSamples;
            if (_session.AccumulatedLength < requiredLen) return;

            float[] waveform = _session.SnapshotAccumulated();
            try { TranscribeAndPublish(waveform, publishToStream: true, ct); }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { throw; }
            catch (Exception ex)
            {
                // ChunkId is normally advanced inside TranscribeAndPublish on a successful
                // pass; if the transcription threw, advance it here so a deterministic
                // failure (e.g. permanent OOM, model error) does NOT busy-spin every
                // subsequent FeedAudio against the same window. The skipped window's
                // audio is still in _accumulated and will be re-encoded by the next
                // successful streaming pass — and definitively by RunFinalTranscription
                // — so accuracy is unaffected; only the partial stream emit is dropped.
                _session.ChunkId += 1;
                _logger.LogWarning(ex,
                    "Qwen3 ASR streaming transcription failed (session: {SessionId}, chunk: {ChunkId}); skipping the partial emit, the audio is retained for the next pass.",
                    _session.SessionId, _session.ChunkId - 1);
            }
        }

        private AsrResult TranscribeAndPublish(float[] waveform, bool publishToStream, CancellationToken ct)
        {
            string? prefix = _runtime.BuildStreamingPrefix(
                _session.ChunkId,
                _accumulatedRaw,
                _session.UnfixedChunkNum,
                _session.UnfixedTokenNum);

            _recognitionGate.Wait(ct);
            Qwen3AsrTranscription transcription;
            try
            {
                ct.ThrowIfCancellationRequested();
                transcription = _runtime.Transcribe(
                    waveform.AsSpan(),
                    _session.Context,
                    _session.Language,
                    prefix,
                    ct,
                    _melState);
                ct.ThrowIfCancellationRequested();
            }
            finally
            {
                _recognitionGate.Release();
            }

            // When we prime the model with a prefix, it only emits the tail. The
            // *cumulative* transcript is prefix + new tail. Likewise the language
            // tag is only emitted on the first uncued pass, so retain the last
            // detected value if this round returned an empty one.
            string fullText = (prefix ?? string.Empty) + transcription.Text;
            string language = string.IsNullOrEmpty(transcription.Language)
                ? _session.LatestLanguage
                : transcription.Language;

            _accumulatedRaw = fullText;
            _session.LatestText = fullText;
            _session.LatestLanguage = language;
            int chunkIndex = _session.ChunkId;
            _session.ChunkId += 1;

            // Update aggregated metrics for the session summary.
            _transcribeCount += 1;
            _totalAudioSeconds += transcription.AudioSeconds;
            _totalInferenceSeconds += transcription.InferenceSeconds;
            _totalEncodeSeconds += transcription.EncodeSeconds;
            _totalPrefillSeconds += transcription.PrefillSeconds;
            _totalDecodeSeconds += transcription.DecodeStepSeconds;

            var result = new AsrResult
            {
                Text = fullText,
                Language = language,
            };

            if (publishToStream)
            {
                bool delivered = _session.PublishResult(result);
                if (delivered && !_firstResultPublished)
                {
                    _firstResultPublished = true;
                    _firstResultElapsedSeconds = _sessionSw.Elapsed.TotalSeconds;
                    _firstResultAudioSeconds = transcription.AudioSeconds;
                    _logger.LogInformation(
                        "Qwen3-ASR 首次流式输出: session={SessionId} 首字耗时={FirstMs:F0}ms (累计音频={AudioSec:F2}s, 本次推理={InfMs:F0}ms, prefill/TTFT={PrefillMs:F0}ms)",
                        _session.SessionId,
                        _firstResultElapsedSeconds * 1000.0,
                        transcription.AudioSeconds,
                        transcription.InferenceSeconds * 1000.0,
                        transcription.PrefillSeconds * 1000.0);
                }

                _logger.LogDebug(
                    "Qwen3-ASR 流式分块完成: session={SessionId} chunk={Chunk} audio={AudioSec:F2}s inf={InfMs:F0}ms RTF={Rtf:F2}",
                    _session.SessionId,
                    chunkIndex,
                    transcription.AudioSeconds,
                    transcription.InferenceSeconds * 1000.0,
                    transcription.AudioSeconds > 0 ? transcription.InferenceSeconds / transcription.AudioSeconds : 0.0);
            }

            return result;
        }
    }

    /// <summary>
    /// Aggregated timing metrics for one Qwen3 ASR session lifecycle. Consumed by the
    /// service to emit a single info-level summary on <c>FinishSessionAsync</c>.
    /// </summary>
    internal readonly record struct Qwen3AsrSessionMetrics(
        double ElapsedSeconds,
        int TranscribeCount,
        double TotalAudioSeconds,
        double TotalInferenceSeconds,
        double TotalEncodeSeconds,
        double TotalPrefillSeconds,
        double TotalDecodeStepSeconds,
        bool FirstResultPublished,
        double FirstResultElapsedSeconds,
        double FirstResultAudioSeconds);
}
