namespace VoiceBox.TTS.DoubaoOnline.Transport
{
    /// <summary>
    /// 对底层双工二进制 WebSocket 通道的抽象。抽出接口是为了让连接 worker 的收发逻辑
    /// 可用 fake 通道做单元测试，无需真实网络。
    /// </summary>
    internal interface IWebSocketChannel : IDisposable
    {
        /// <summary>握手成功后服务端返回的 logid (X-Tt-Logid)，用于排障；可能为 null。</summary>
        string? LogId { get; }

        /// <summary>建立连接并完成握手。失败抛异常。</summary>
        Task ConnectAsync(CancellationToken ct);

        /// <summary>发送一整条二进制消息 (EndOfMessage=true)。</summary>
        Task SendAsync(ReadOnlyMemory<byte> frame, CancellationToken ct);

        /// <summary>
        /// 接收一整条二进制消息 (自动重组分片)。返回 null 表示对端已正常关闭 (收到 Close 帧)。
        /// </summary>
        Task<byte[]?> ReceiveAsync(CancellationToken ct);

        /// <summary>尽力发送 Close 帧并关闭连接 (不抛出)。</summary>
        Task CloseAsync(CancellationToken ct);
    }

    /// <summary>创建 <see cref="IWebSocketChannel"/> 的工厂 (可注入以替换为 fake)。</summary>
    internal interface IWebSocketChannelFactory
    {
        IWebSocketChannel Create(Uri uri, IReadOnlyDictionary<string, string> headers, int connectTimeoutSec);
    }
}
