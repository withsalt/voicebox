using System.Text;
using System.Text.Json;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// VoxCPM 文本分词器: tokenizer.json 的 SentencePiece 风格 BPE 的自包含端口
    /// (Microsoft.ML.Tokenizers 2.0 的 LlamaTokenizer 只吃 .model 文件, 且无法复刻
    /// 下述定制逻辑, 故此处直接按 tokenizer.json 语义实现)。
    ///
    /// 语义 (与 python `mask_multichar_chinese_tokens(LlamaTokenizerFast)` 逐 id 对拍):
    ///   1. normalizer: Prepend("▁") + Replace(" "→"▁"); 无 pre_tokenizer;
    ///   2. 经典 BPE: 每轮合并 rank 最小的相邻符号对;
    ///   3. byte_fallback: 不在词表的符号逐 utf-8 字节替换为 &lt;0xXX&gt; token;
    ///   4. **不加 BOS** (参考实现走 tokenize(), 绕过 post_processor);
    ///   5. 多字中文 token 拆单字: 产出 token 若命中 multichar_zh.json 映射, 替换为逐字 id 序列。
    ///
    /// 算法与 huggingface tokenizers 库的一致性已在导出侧用 7 组中英混合用例验证。
    /// </summary>
    internal sealed class VoxCpmTokenizer
    {
        private readonly Dictionary<string, int> _vocab;
        private readonly Dictionary<(string, string), int> _mergeRanks;
        private readonly Dictionary<int, int[]> _zhSplit;
        private readonly int[] _byteIds = new int[256];

        public VoxCpmTokenizer(string tokenizerJsonPath, string multicharZhJsonPath)
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
            var model = doc.RootElement.GetProperty("model");
            if (model.GetProperty("type").GetString() != "BPE")
                throw new InvalidDataException("tokenizer.json 不是 BPE 模型。");

            var vocabEl = model.GetProperty("vocab");
            _vocab = new Dictionary<string, int>(vocabEl.GetPropertyCount(), StringComparer.Ordinal);
            foreach (var p in vocabEl.EnumerateObject())
                _vocab[p.Name] = p.Value.GetInt32();

            var mergesEl = model.GetProperty("merges");
            _mergeRanks = new Dictionary<(string, string), int>(mergesEl.GetArrayLength());
            int rank = 0;
            foreach (var m in mergesEl.EnumerateArray())
            {
                // 两种序列化: "a b" 字符串或 ["a","b"] 数组
                string left, right;
                if (m.ValueKind == JsonValueKind.String)
                {
                    string s = m.GetString()!;
                    int sp = s.IndexOf(' ');
                    left = s[..sp];
                    right = s[(sp + 1)..];
                }
                else
                {
                    left = m[0].GetString()!;
                    right = m[1].GetString()!;
                }
                _mergeRanks.TryAdd((left, right), rank++);
            }

            for (int b = 0; b < 256; b++)
            {
                if (!_vocab.TryGetValue($"<0x{b:X2}>", out _byteIds[b]))
                    throw new InvalidDataException($"tokenizer.json 缺少 byte-fallback token <0x{b:X2}>。");
            }

            var zh = JsonSerializer.Deserialize<Dictionary<string, int[]>>(File.ReadAllText(multicharZhJsonPath))
                     ?? throw new InvalidDataException($"multichar_zh.json 解析失败: {multicharZhJsonPath}");
            _zhSplit = new Dictionary<int, int[]>(zh.Count);
            foreach (var (k, v) in zh)
                _zhSplit[int.Parse(k)] = v;
        }

        public int VocabSize => _vocab.Count;

        /// <summary>tokenize 文本 (无 BOS, 含多字中文拆分)。</summary>
        public int[] Encode(string text)
        {
            var raw = EncodeBpe(text);
            // 多字中文拆单字 (按产出 id 查预计算映射表)
            List<int>? expanded = null;
            for (int i = 0; i < raw.Count; i++)
            {
                if (_zhSplit.TryGetValue(raw[i], out int[]? split))
                {
                    if (expanded is null)
                    {
                        expanded = new List<int>(raw.Count + 8);
                        for (int j = 0; j < i; j++) expanded.Add(raw[j]);
                    }
                    expanded.AddRange(split);
                }
                else
                {
                    expanded?.Add(raw[i]);
                }
            }
            return expanded is null ? [.. raw] : [.. expanded];
        }

        private List<int> EncodeBpe(string text)
        {
            // normalizer: Prepend("▁") + Replace(" "→"▁")
            string s = "▁" + text.Replace(' ', '▁');

            // 初始符号 = unicode 码点 (含代理对成对处理)
            var symbols = new List<string>(s.Length);
            for (int i = 0; i < s.Length; i++)
            {
                if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                {
                    symbols.Add(s.Substring(i, 2));
                    i++;
                }
                else
                {
                    symbols.Add(s[i].ToString());
                }
            }

            // 经典贪心 BPE: 每轮找全局 rank 最小的相邻对合并
            while (symbols.Count > 1)
            {
                int bestRank = int.MaxValue, bestIdx = -1;
                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    if (_mergeRanks.TryGetValue((symbols[i], symbols[i + 1]), out int r) && r < bestRank)
                    {
                        bestRank = r;
                        bestIdx = i;
                    }
                }
                if (bestIdx < 0) break;
                symbols[bestIdx] += symbols[bestIdx + 1];
                symbols.RemoveAt(bestIdx + 1);
            }

            var ids = new List<int>(symbols.Count);
            foreach (string sym in symbols)
            {
                if (_vocab.TryGetValue(sym, out int id))
                {
                    ids.Add(id);
                }
                else
                {
                    // byte_fallback: 逐 utf-8 字节
                    foreach (byte b in Encoding.UTF8.GetBytes(sym))
                        ids.Add(_byteIds[b]);
                }
            }
            return ids;
        }
    }
}
