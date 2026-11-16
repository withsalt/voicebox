using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.TTS.DoubaoOnline.Protocol;
using VoiceBox.TTS.DoubaoOnline.Transport;

namespace VoiceBox.TTS.DoubaoOnline.Session
{
    /// <summary>
    /// 驱动单次合成会话的 WebSocket 收发：
    ///   * 上行 (drive)：StartConnection → StartSession → TaskRequest(可多次) → FinishSession；
    ///     所有上行发送经 <see cref="_sendLock"/> 串行化 (WebSocket 不支持并发发送)。
    ///   * 下行 (receive loop)：解析事件帧，推进握手门 / 推送音频块 / 收集字幕计费；
    ///     收到 SessionFinished 结束音频流，收到失败事件或 error 帧以异常收尾。
    /// </summary>
    internal sealed class SeedTtsConnectionWorker
    {
        private readonly SeedTtsSession _session;
        private readonly IWebSocketChannel _channel;
        private readonly DoubaoOnlineTTSOptions _options;
        private readonly string _startSessionJson;
        private readonly ILogger _logger;
        private readonly SemaphoreSlim _sendLock = new(1, 1);

        private Task? _receiveTask;

        public SeedTtsConnectionWorker(
            SeedTtsSession session,
            IWebSocketChannel channel,
            DoubaoOnlineTTSOptions options,
            string startSessionJson,
            ILogger? logger = null)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _channel = channel ?? throw new ArgumentNullException(nameof(channel));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _startSessionJson = startSessionJson ?? throw new ArgumentNullException(nameof(startSessionJson));
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>
        /// 建连、启动接收 loop，完成 StartConnection / StartSession 握手 (等待对应下行事件)。
        /// 任一步骤失败会抛出，交由上层回滚会话并释放资源。
        /// </summary>
        public async Task StartAsync(CancellationToken ct)
        {
            await _channel.ConnectAsync(ct).ConfigureAwait(false);
            _session.LogId = _channel.LogId;

            _receiveTask = Task.Run(() => ReceiveLoopAsync(_session.CancellationToken));

            // StartConnection → 等 ConnectionStarted
            await SendEventAsync(EventType.StartConnection, SeedTtsRequestBuilder.EmptyJson, ct).ConfigureAwait(false);
            await AwaitHandshakeAsync(_session.ConnectionStarted, "ConnectionStarted", ct).ConfigureAwait(false);

            // StartSession → 等 SessionStarted
            await SendEventAsync(EventType.StartSession, _startSessionJson, ct).ConfigureAwait(false);
            await AwaitHandshakeAsync(_session.SessionStarted, "SessionStarted", ct).ConfigureAwait(false);
        }

        /// <summary>
        /// 等待一个握手门完成，带连接超时保护：<see cref="DoubaoOnlineTTSOptions.ConnectTimeoutSec"/> 内
        /// 未收到对应下行事件则抛超时 (避免服务端建连后静默导致挂起)。门本身也可能以服务端错误收尾。
        /// </summary>
        private async Task AwaitHandshakeAsync(Task gate, string what, CancellationToken ct)
        {
            if (_options.ConnectTimeoutSec > 0)
            {
                try { await gate.WaitAsync(TimeSpan.FromSeconds(_options.ConnectTimeoutSec), ct).ConfigureAwait(false); }
                catch (TimeoutException)
                {
                    throw new DoubaoOnlineTTSException(
                        $"Timed out after {_options.ConnectTimeoutSec}s waiting for {what} (session {_session.SessionId}).");
                }
            }
            else
            {
                await gate.WaitAsync(ct).ConfigureAwait(false);
            }
        }

        /// <summary>发送一段待合成文本 (TaskRequest)。可多次调用以流式追加文本。</summary>
        public Task SendTextAsync(string text, CancellationToken ct)
        {
            string json = SeedTtsRequestBuilder.BuildTaskRequestJson(text);
            return SendEventAsync(EventType.TaskRequest, json, ct);
        }

        /// <summary>发送 FinishSession，通知服务端本次合成的文本已发完。不等待收尾 (由接收 loop 处理)。</summary>
        public Task FinishSessionAsync(CancellationToken ct)
            => SendEventAsync(EventType.FinishSession, SeedTtsRequestBuilder.EmptyJson, ct);

        /// <summary>等待接收 loop 自然结束 (SessionFinished / 通道关闭)，随后发 FinishConnection 并优雅关闭。</summary>
        public async Task StopAsync(CancellationToken ct)
        {
            try
            {
                if (_receiveTask is not null)
                    await _receiveTask.WaitAsync(ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                _logger.LogWarning("SeedTTS worker did not drain before timeout (session {SessionId}).", _session.SessionId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "SeedTTS worker faulted on stop (session {SessionId}).", _session.SessionId);
            }
            finally
            {
                try { await SendEventAsync(EventType.FinishConnection, SeedTtsRequestBuilder.EmptyJson, CancellationToken.None).ConfigureAwait(false); }
                catch { /* best-effort */ }
                try { await _channel.CloseAsync(CancellationToken.None).ConfigureAwait(false); } catch { }
                _channel.Dispose();
                _sendLock.Dispose();
            }
        }

        /// <summary>强制中止 (会话取消/服务释放)：取消 token 使接收 loop 退出，随后关闭通道。</summary>
        public async Task AbortAsync()
        {
            _session.Cancel();
            if (_receiveTask is not null)
            {
                try { await _receiveTask.ConfigureAwait(false); } catch { }
            }
            try { await _channel.CloseAsync(CancellationToken.None).ConfigureAwait(false); } catch { }
            _channel.Dispose();
            _sendLock.Dispose();
        }

        private async Task SendEventAsync(EventType evt, string json, CancellationToken ct)
        {
            byte[] frame = SeedTtsMessageCodec.EncodeClientEvent(evt, _session.SessionId, json, _options.CompressPayload);
            await _sendLock.WaitAsync(ct).ConfigureAwait(false);
            try { await _channel.SendAsync(frame, ct).ConfigureAwait(false); }
            finally { _sendLock.Release(); }
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
                    catch (OperationCanceledException) { break; }

                    if (raw is null)
                    {
                        // 对端关闭。若会话尚未正常结束, 视为异常收尾。
                        if (!_session.SessionFinished.IsCompleted)
                            _session.TrySetError(new DoubaoOnlineTTSException("Connection closed before session finished."));
                        break;
                    }

                    ServerMessage msg = SeedTtsMessageCodec.Decode(raw);
                    if (Dispatch(msg)) break; // 返回 true 表示会话已收尾, 退出接收
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                _logger.LogError(ex, "SeedTTS receive loop faulted (session {SessionId}).", _session.SessionId);
                _session.TrySetError(new DoubaoOnlineTTSException("Receive loop faulted.", ex));
            }
        }

        /// <summary>处理一条下行消息。返回 true 表示会话已收尾 (正常/失败)，接收 loop 应退出。</summary>
        private bool Dispatch(ServerMessage msg)
        {
            if (msg.IsError)
            {
                var ex = new SeedTtsServerException(msg.ErrorCode, msg.ErrorMessage ?? "unknown", _session.LogId);
                _logger.LogError("SeedTTS server error (session {SessionId}): {Error}", _session.SessionId, ex.Message);
                _session.TrySetError(ex);
                return true;
            }

            if (msg.IsAudio)
            {
                if (msg.AudioPayload is { Length: > 0 })
                    _session.PublishAudio(PcmConverter.Int16LittleEndianToFloat(msg.AudioPayload));
                return false;
            }

            switch (msg.Event)
            {
                case EventType.ConnectionStarted:
                    _session.MarkConnectionStarted();
                    return false;

                case EventType.SessionStarted:
                    _session.MarkSessionStarted();
                    return false;

                case EventType.TtsSentenceStart:
                case EventType.TtsSentenceEnd:
                case EventType.TtsResponse:
                case EventType.TtsEnded:
                    // 句起/句终/结束等元数据。音频经 AudioOnlyServer 帧下发, 这里无需处理。
                    return false;

                case EventType.TtsSubtitle:
                    CaptureSubtitle(msg.Json);
                    return false;

                case EventType.UsageResponse:
                case EventType.SessionFinished:
                    CaptureUsage(msg.Json);
                    if (msg.Event == EventType.SessionFinished)
                    {
                        _session.MarkSessionFinished();
                        return true;
                    }
                    return false;

                case EventType.ConnectionFailed:
                case EventType.SessionFailed:
                    _session.TrySetError(BuildFailure(msg));
                    return true;

                default:
                    _logger.LogDebug("SeedTTS unhandled event {Event} (session {SessionId}).", msg.Event, _session.SessionId);
                    return false;
            }
        }

        private void CaptureSubtitle(string? json)
        {
            if (string.IsNullOrEmpty(json)) return;
            try
            {
                var payload = SeedTtsPayloadSerializer.DeserializeResponse(json!);
                if (payload?.Words is { Count: > 0 } words)
                    _session.AddSubtitles(words);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "SeedTTS failed to parse subtitle payload (session {SessionId}).", _session.SessionId);
            }
        }

        private void CaptureUsage(string? json)
        {
            if (string.IsNullOrEmpty(json)) return;
            try
            {
                var payload = SeedTtsPayloadSerializer.DeserializeResponse(json!);
                if (payload?.Usage is { } usage)
                    _session.SetUsage(usage.TextWords);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "SeedTTS failed to parse usage payload (session {SessionId}).", _session.SessionId);
            }
        }

        private SeedTtsServerException BuildFailure(ServerMessage msg)
        {
            string message = msg.Event.ToString();
            if (!string.IsNullOrEmpty(msg.Json))
            {
                var err = SeedTtsPayloadSerializer.TryDeserializeError(msg.Json!);
                string detail = err?.Message ?? err?.Error ?? msg.Json!;
                message = $"{msg.Event}: {detail}";
            }
            _logger.LogError("SeedTTS session failed (session {SessionId}): {Message}", _session.SessionId, message);
            return new SeedTtsServerException(0, message, _session.LogId);
        }
    }
}
