using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// 文本嵌入查表 (embeddings/text_embedding.bin, [V,H] fp32 little-endian)。
    /// 查表留在宿主让 lm.onnx 保持纯 MatMul (int4 量化友好), 与 qwen3-tts 同构。
    /// 按需 RandomAccess 逐行读 (每次合成只查几十~几百行 × 4KB), 不把 ~300MB 整表常驻内存。
    /// </summary>
    internal sealed class EmbeddingStore : IDisposable
    {
        private readonly SafeFileHandle _file;
        private readonly int _vocab;
        private readonly int _hidden;

        public EmbeddingStore(string binPath, int vocab, int hidden)
        {
            _vocab = vocab;
            _hidden = hidden;
            long expect = (long)vocab * hidden * sizeof(float);
            var fi = new FileInfo(binPath);
            if (!fi.Exists)
                throw new FileNotFoundException($"缺少文本嵌入表: {binPath}");
            if (fi.Length != expect)
                throw new InvalidDataException(
                    $"text_embedding.bin 大小不符: 期望 {expect} 字节 ([{vocab},{hidden}] fp32), 实际 {fi.Length}。");
            _file = File.OpenHandle(binPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        }

        /// <summary>把 token id 序列展开成 [T*H] 行主序嵌入。</summary>
        public float[] Lookup(ReadOnlySpan<int> ids)
        {
            var outp = new float[ids.Length * _hidden];
            var bytes = MemoryMarshal.AsBytes(outp.AsSpan());
            int rowBytes = _hidden * sizeof(float);
            for (int i = 0; i < ids.Length; i++)
            {
                int id = ids[i];
                if ((uint)id >= (uint)_vocab)
                    throw new ArgumentOutOfRangeException(nameof(ids), $"token id {id} 超出词表 [0,{_vocab})。");
                var row = bytes.Slice(i * rowBytes, rowBytes);
                long ofs = (long)id * rowBytes;
                int done = 0;
                while (done < rowBytes)
                {
                    int n = RandomAccess.Read(_file, row[done..], ofs + done);
                    if (n <= 0) throw new EndOfStreamException("text_embedding.bin 读取意外到达文件尾。");
                    done += n;
                }
            }
            return outp;
        }

        public void Dispose() => _file.Dispose();
    }
}
