namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 加载 sherpa-onnx KWS 模型自带的 en.phone (CMU 风格发音词典: "WORD  PH1 PH2 ..."),
    /// 用于把英文关键词逐词转换为模型 tokens.txt 里的音素序列。
    /// 词典懒加载且按 OrdinalIgnoreCase 索引, 单例可被多个 KwsService 共享。
    /// </summary>
    internal sealed class CmuPhoneDictionary
    {
        private static readonly char[] _whitespace = { ' ', '\t' };

        private readonly Dictionary<string, string[]> _word2Phones;

        public int Count => _word2Phones.Count;

        public CmuPhoneDictionary(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("CMU phone file path must not be empty.", nameof(filePath));
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"CMU phone file not found: {filePath}", filePath);

            _word2Phones = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            foreach (var line in File.ReadAllLines(filePath))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                if (line[0] == ';' || line[0] == '#') continue;

                var parts = line.Split(_whitespace, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2) continue;

                string word = parts[0];
                // CMUdict 用 "(2)" 标记同形异读; 只保留默认读音, 后续读音忽略。
                int paren = word.IndexOf('(');
                if (paren > 0) continue;

                if (_word2Phones.ContainsKey(word)) continue;

                var phones = new string[parts.Length - 1];
                Array.Copy(parts, 1, phones, 0, phones.Length);
                _word2Phones[word] = phones;
            }
        }

        public bool TryGet(string word, out string[] phones) => _word2Phones.TryGetValue(word, out phones!);
    }
}
