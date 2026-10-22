using System.IO.Compression;

namespace VoiceBox.TTS.DoubaoOnline.Protocol
{
    /// <summary>
    /// Gzip 压缩/解压工具。上行 JSON payload 可选 Gzip 压缩 (由 <see cref="DoubaoOnlineTTSOptions.CompressPayload"/>
    /// 控制)；下行帧按 header 中的压缩位决定是否解压。
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
