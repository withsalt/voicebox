using System.IO.Compression;

namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// Gzip 压缩/解压工具。协议中 full client request 与 audio only request 的 payload
    /// 默认使用 Gzip 压缩，服务端会以相同方式压缩响应。
    /// </summary>
    internal static class GzipCodec
    {
        public static byte[] Compress(ReadOnlySpan<byte> data)
        {
            using var output = new MemoryStream(Math.Max(64, data.Length / 2));
            using (var gzip = new GZipStream(output, CompressionLevel.Fastest, leaveOpen: true))
            {
                gzip.Write(data);
            }
            return output.ToArray();
        }

        public static byte[] Decompress(ReadOnlySpan<byte> data)
        {
            using var input = new MemoryStream(data.ToArray());
            using var gzip = new GZipStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream(Math.Max(256, data.Length * 4));
            gzip.CopyTo(output);
            return output.ToArray();
        }
    }
}
