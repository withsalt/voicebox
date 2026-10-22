using System.Buffers;
using System.Net.WebSockets;

namespace VoiceBox.ASR.DoubaoOnline.Transport
{
    /// <summary>
    /// 基于 <see cref="ClientWebSocket"/> 的生产实现。负责握手鉴权头注入、分片重组与优雅关闭。
    /// </summary>
    internal sealed class ClientWebSocketChannel : IWebSocketChannel
    {
        private const int ReceiveBufferSize = 16 * 1024;

        private readonly Uri _uri;
        private readonly IReadOnlyDictionary<string, string> _headers;
        private readonly int _connectTimeoutSec;
        private readonly ClientWebSocket _socket = new();
        private int _disposed;

        public string? LogId { get; private set; }

        public ClientWebSocketChannel(Uri uri, IReadOnlyDictionary<string, string> headers, int connectTimeoutSec)
        {
            _uri = uri ?? throw new ArgumentNullException(nameof(uri));
            _headers = headers ?? throw new ArgumentNullException(nameof(headers));
            _connectTimeoutSec = connectTimeoutSec;
        }

        public async Task ConnectAsync(CancellationToken ct)
        {
            _socket.Options.CollectHttpResponseDetails = true;
            foreach (var kv in _headers)
                _socket.Options.SetRequestHeader(kv.Key, kv.Value);

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            if (_connectTimeoutSec > 0)
                timeoutCts.CancelAfter(TimeSpan.FromSeconds(_connectTimeoutSec));

            await _socket.ConnectAsync(_uri, timeoutCts.Token).ConfigureAwait(false);

            // 记录服务端 logid，便于排障 (强烈建议)。
            if (_socket.HttpResponseHeaders is { } respHeaders &&
                respHeaders.TryGetValue("X-Tt-Logid", out var values))
            {
                LogId = values.FirstOrDefault();
            }
        }

        public Task SendAsync(ReadOnlyMemory<byte> frame, CancellationToken ct)
            => _socket.SendAsync(frame, WebSocketMessageType.Binary, endOfMessage: true, ct).AsTask();

        public async Task<byte[]?> ReceiveAsync(CancellationToken ct)
        {
            var rented = ArrayPool<byte>.Shared.Rent(ReceiveBufferSize);
            try
            {
                using var ms = new MemoryStream();
                while (true)
                {
                    WebSocketReceiveResult result = await _socket
                        .ReceiveAsync(new ArraySegment<byte>(rented), ct)
                        .ConfigureAwait(false);

                    if (result.MessageType == WebSocketMessageType.Close)
                        return null;

                    ms.Write(rented, 0, result.Count);
                    if (result.EndOfMessage) break;
                }
                return ms.ToArray();
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }

        public async Task CloseAsync(CancellationToken ct)
        {
            try
            {
                if (_socket.State is WebSocketState.Open or WebSocketState.CloseReceived)
                {
                    await _socket
                        .CloseAsync(WebSocketCloseStatus.NormalClosure, "done", ct)
                        .ConfigureAwait(false);
                }
            }
            catch { /* best-effort close */ }
        }

        public void Abort() => _socket.Abort();

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            try { _socket.Dispose(); } catch { }
        }
    }

    internal sealed class ClientWebSocketChannelFactory : IWebSocketChannelFactory
    {
        public IWebSocketChannel Create(Uri uri, IReadOnlyDictionary<string, string> headers, int connectTimeoutSec)
            => new ClientWebSocketChannel(uri, headers, connectTimeoutSec);
    }
}
