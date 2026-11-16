namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 符号表 (token <-> id), 对应 sherpa-onnx 的 SymbolTable。
    /// tokens.txt 每行格式为 "&lt;symbol&gt; &lt;id&gt;", 例如 "&lt;blk&gt; 0"。
    /// </summary>
    internal sealed class SymbolTable
    {
        private readonly Dictionary<string, int> _sym2Id;
        private readonly Dictionary<int, string> _id2Sym;

        public int Count => _id2Sym.Count;

        public SymbolTable(string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("Tokens file path must not be empty.", nameof(filePath));
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"Tokens file not found: {filePath}", filePath);

            _sym2Id = new Dictionary<string, int>(StringComparer.Ordinal);
            _id2Sym = new Dictionary<int, string>();

            using var reader = new StreamReader(filePath);
            string? line;
            int lineNo = 0;
            while ((line = reader.ReadLine()) != null)
            {
                lineNo++;
                if (line.Length == 0) continue;

                // 与 sherpa-onnx 一致: 第一个空白前为 symbol, 之后为 id。symbol 中可以包含 unicode 字符,
                // 但不能包含空白。最后一个 token 是 id。
                int lastSpace = line.LastIndexOf(' ');
                if (lastSpace <= 0)
                    throw new InvalidDataException($"Malformed tokens.txt at line {lineNo}: '{line}'");

                string sym = line.Substring(0, lastSpace);
                string idText = line.Substring(lastSpace + 1).Trim();
                if (!int.TryParse(idText, out int id))
                    throw new InvalidDataException($"Malformed tokens.txt at line {lineNo}: id '{idText}' is not an integer.");

                _sym2Id[sym] = id;
                _id2Sym[id] = sym;
            }
        }

        public bool Contains(string sym) => _sym2Id.ContainsKey(sym);

        public int GetId(string sym)
        {
            if (!_sym2Id.TryGetValue(sym, out int id))
                throw new KeyNotFoundException($"Symbol '{sym}' not found in tokens.txt.");
            return id;
        }

        public bool TryGetId(string sym, out int id) => _sym2Id.TryGetValue(sym, out id);

        public string GetSymbol(int id) => _id2Sym.TryGetValue(id, out string? sym) ? sym : string.Empty;
    }
}
