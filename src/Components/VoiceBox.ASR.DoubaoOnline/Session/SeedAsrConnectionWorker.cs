using System.Buffers;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.ASR.DoubaoOnline.Protocol;
using VoiceBox.ASR.DoubaoOnline.Transport;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.DoubaoOnline.Session
{
    /// <summary>
    /// 驱动单条会话的 WebSocket 收发循环：
    ///   * 发送 loop：从会话音频通道取 float 块 → 累积成 PCM 帧 → (可选 Gzip) 发送 audio only request；
    ///     会话进入 Finishing 后冲刷剩余并发送最后一包 (负包)。
    ///   * 接收 loop：重组并解码下行帧 → 映射为 <see cref="AsrResult"/> 推送给消费者；
    ///     收到最后一包置最终结果，收到 error 帧以异常收尾。
    /// </summary>
    internal sealed class SeedAsrConnectionWorker
    {
        private readonly SeedAsrSession _session;
        private readonly IWebSocketChannel _channel;
        private readonly DoubaoOnlineASROptions _options;
        private readonly string _requestJson;
        private readonly string _language;
        private readonly ILogger _logger;
        private readonly CancellationTokenSource _transportStop;

        private readonly int _frameSamples;
        private Task? _sendTask;
        private Task? _receiveTask;

        public SeedAsrConnectionWorker(
            SeedAsrSession session,
            IWebSocketChannel channel,
            DoubaoOnlineASROptions options,
            string requestJson,
            ILogger? logger = null)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _channel = channel ?? throw new ArgumentNullException(nameof(channel));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _requestJson = requestJson ?? throw new ArgumentNullException(nameof(requestJson));
            _logger = logger ?? NullLogger.Instance;
            _transportStop = CancellationTokenSource.CreateLinkedTokenSource(session.CancellationToken);
            _language = string.Empty;
            _frameSamples = Math.Max(1, checked((int)((long)options.SampleRate * options.Channels * options.PackageDurationMs / 1000)));
        }

        /// <summary>
        /// 建连、发送 full client request，随后启动收发后台循环。
        /// 连接或首包发送失败会直接抛出，交由上层回滚会话计数并释放资源。
        /// </summary>
        public async Task StartAsync(CancellationToken ct)
        {
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct, _session.CancellationToken);
            ct = linked.Token;
            using var registration = ct.Register(_channel.Abort);
            ct.ThrowIfCancellationRequested();
            await _channel.ConnectAsync(ct).ConfigureAwait(false);
            _session.LogId = _channel.LogId;

            byte[] first = SeedAsrMessageCodec.EncodeFullClientRequest(_requestJson, _options.CompressAudio);
            await _channel.SendAsync(first, ct).ConfigureAwait(false);
            ct.ThrowIfCancellationRequested();

            _receiveTask = Task.Run(() => ReceiveLoopAsync(_transportStop.Token));
            _sendTask = Task.Run(() => SendLoopAsync(_transportStop.Token));
        }

        /// <summary>等待收发循环自然结束并关闭连接；通道释放由会话清理所有者负责。</summary>
        public async Task StopAsync(CancellationToken ct)
        {
            // The stop budget is an internal transport-drain timeout. Aborting the channel
            // must not mark the session cancelled, otherwise a final result already received
            // would be discarded by FinishCoreAsync after a close timeout.
            using var registration = ct.Register(AbortTransport);
            try
            {
                var pending = new List<Task>(2);
                if (_sendTask is not null) pending.Add(_sendTask);
                if (_receiveTask is not null) pending.Add(_receiveTask);
                if (pending.Count > 0)
                    await Task.WhenAll(pending).WaitAsync(ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                _logger.LogWarning("SeedASR worker did not drain before timeout (session {SessionId}).", _session.SessionId);
                AbortTransport();
                await AwaitLoopsAsync().ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "SeedASR worker faulted on stop (session {SessionId}).", _session.SessionId);
            }
            finally
            {
                if (ct.IsCancellationRequested)
                {
                    AbortTransport();
                }
                else
                {
                    try { await _channel.CloseAsync(ct).ConfigureAwait(false); }
                    catch { AbortTransport(); }
                }
            }
        }

        /// <summary>强制中止连接，等待收发循环退出后释放通道。</summary>
        public async Task AbortAsync()
        {
            SignalAbort();
            await AwaitLoopsAsync().ConfigureAwait(false);
            try { _channel.Dispose(); }
            finally { _transportStop.Dispose(); }
        }

        public void SignalAbort()
        {
            _session.AbortWork();
            AbortTransport();
        }

        internal void AbortTransport()
        {
            // Audio ingress faults can race with terminal cleanup after the
            // failure has woken a Finish waiter. Aborting is safe after release.
            try { _transportStop.Cancel(); }
            catch (ObjectDisposedException) { return; }
            try { _channel.Abort(); }
            catch (ObjectDisposedException) { }
        }

        private async Task AwaitLoopsAsync()
        {
            var tasks = new List<Task>(2);
            if (_sendTask is not null) tasks.Add(_sendTask);
            if (_receiveTask is not null) tasks.Add(_receiveTask);
            try { await Task.WhenAll(tasks).ConfigureAwait(false); } catch { }
        }

        // ---- 发送循环 ----

        private async Task SendLoopAsync(CancellationToken ct)
        {
            // 累积 float 采样, 凑满一帧 (frameSamples) 再转 PCM16 发送; 会话结束时冲刷剩余作为最后一包。
            float[] acc = ArrayPool<float>.Shared.Rent(_frameSamples * 2);
            int accLen = 0;
            var clock = Stopwatch.StartNew();
            long samplesSent = 0;

            try
            {
                while (true)
                {
                    bool more;
                    try { more = await _session.WaitToReadAudioAsync(ct).ConfigureAwait(false); }
                    catch (OperationCanceledException) when (ct.IsCancellationRequested) { break; }

                    if (!more) break; // 音频写端 complete: 进入冲刷收尾

                    while (!ct.IsCancellationRequested && _session.TryReadAudio(out var pooled))
                    {
                        try
                        {
                            ReadOnlySpan<float> span = pooled.Span;
                            EnsureAcc(ref acc, accLen + span.Length);
                            span.CopyTo(acc.AsSpan(accLen));
                            accLen += span.Length;
                        }
                        finally { pooled.Dispose(); }

                        while (accLen >= _frameSamples)
                        {
                            byte[] frame = BuildAudioFrame(acc.AsSpan(0, _frameSamples), isLast: false);
                            await _channel.SendAsync(frame, ct).ConfigureAwait(false);
                            int rest = accLen - _frameSamples;
                            acc.AsSpan(_frameSamples, rest).CopyTo(acc);
                            accLen = rest;

                            samplesSent += _frameSamples;
                            await ThrottleAsync(clock, samplesSent, ct).ConfigureAwait(false);
                        }
                    }
                }

                // 冲刷剩余 + 发送最后一包 (负包)。即使无剩余也要发一个空的最后一包通知服务端收尾。
                ct.ThrowIfCancellationRequested();
                byte[] lastFrame = BuildAudioFrame(acc.AsSpan(0, accLen), isLast: true);
                await _channel.SendAsync(lastFrame, ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { }
            catch (Exception ex)
            {
                _ = _session.FailAsync(new DoubaoOnlineASRException("Send loop faulted.", ex));
                try { _logger.LogWarning(ex, "SeedASR send loop faulted (session {SessionId}).", _session.SessionId); }
                catch { /* The transport failure already owns session cleanup. */ }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(acc);
            }
        }

        private byte[] BuildAudioFrame(ReadOnlySpan<float> samples, bool isLast)
        {
            byte[] pcm = samples.IsEmpty ? Array.Empty<byte>() : PcmConverter.ToInt16LittleEndian(samples);
            return SeedAsrMessageCodec.EncodeAudioRequest(pcm, _options.CompressAudio, isLast);
        }

        private async Task ThrottleAsync(Stopwatch clock, long samplesSent, CancellationToken ct)
        {
            if (!_options.ThrottleSendToRealtime) return;
            double audioMs = samplesSent * 1000.0 / (_options.SampleRate * Math.Max(1, _options.Channels));
            double elapsedMs = clock.Elapsed.TotalMilliseconds;
            double aheadMs = audioMs - elapsedMs;
            // 只有当发送快于实时时才补 sleep; 麦克风实时流几乎不会触发。留 20ms 容差避免过度抖动。
            if (aheadMs > 20)
                await Task.Delay((int)aheadMs, ct).ConfigureAwait(false);
        }

        private static void EnsureAcc(ref float[] acc, int required)
        {
            if (acc.Length >= required) return;
            int newSize = Math.Max(acc.Length * 2, required);
            var bigger = ArrayPool<float>.Shared.Rent(newSize);
            acc.CopyTo(bigger, 0);
            ArrayPool<float>.Shared.Return(acc);
            acc = bigger;
        }

        // ---- 接收循环 ----

        private async Task ReceiveLoopAsync(CancellationToken ct)
        {
            try
            {
                while (!ct.IsCancellationRequested)
                {
                    byte[]? raw;
                    try { raw = await _channel.ReceiveAsync(ct).ConfigureAwait(false); }
                    catch (OperationCanceledException) when (ct.IsCancellationRequested) { break; }
                    ct.ThrowIfCancellationRequested();

                    if (raw is null)
                    {
                        // 对端关闭。若最终结果尚未产生, 用最近一次结果收尾, 否则视为异常。
                        if (!_session.FinalResult.IsCompleted)
                            _ = _session.FailAsync(new DoubaoOnlineASRException("Connection closed before final result."));
                        break;
                    }

                    ServerMessage msg = SeedAsrMessageCodec.Decode(raw);
                    if (msg.Type == ServerMessageType.Error)
                    {
                        var ex = new SeedAsrServerException(msg.ErrorCode, msg.ErrorMessage ?? "unknown", _session.LogId);
                        _ = _session.FailAsync(ex);
                        try { _logger.LogError("SeedASR server error (session {SessionId}): {Error}", _session.SessionId, ex.Message); }
                        catch { /* Preserve the server error if logging fails. */ }
                        break;
                    }

                    AsrResult result = MapResponse(msg.PayloadJson);
                    _session.PublishResult(result);

                    if (msg.IsLastPackage)
                    {
                        _session.TrySetFinalResult(result);
                        break;
                    }
                }
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested) { }
            catch (Exception ex)
            {
                _ = _session.FailAsync(new DoubaoOnlineASRException("Receive loop faulted.", ex));
                try { _logger.LogError(ex, "SeedASR receive loop faulted (session {SessionId}).", _session.SessionId); }
                catch { /* The transport failure already owns session cleanup. */ }
            }
        }

        private AsrResult MapResponse(string? json)
        {
            if (string.IsNullOrEmpty(json))
                return new AsrResult { Language = _language };
            try
            {
                var payload = SeedAsrPayloadSerializer.DeserializeResponse(json!);
                return payload is null
                    ? new AsrResult { Language = _language }
                    : SeedAsrResultMapper.Map(payload, _language);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse SeedASR response payload (session {SessionId}).", _session.SessionId);
                return new AsrResult { Language = _language };
            }
        }
    }
}
