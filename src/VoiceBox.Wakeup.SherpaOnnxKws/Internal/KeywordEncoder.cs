using VoiceBox.Wakeup.SherpaOnnxKws.Models;

namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// 解析关键词描述, 把每行 "tok1 tok2 ... :score #threshold @phrase" 编码成 token id 列表 +
    /// 可选的 boost / threshold / 原文。与 sherpa-onnx 的 EncodeKeywords 行为一致。
    /// </summary>
    internal static class KeywordEncoder
    {
        private static readonly char[] _whitespaceSeparators = { ' ', '\t' };

        public readonly struct EncodedKeyword
        {
            public readonly int[] TokenIds;
            public readonly float BoostScore;
            public readonly float Threshold;
            public readonly string Phrase;
            public readonly int DebounceMs;

            public EncodedKeyword(int[] tokenIds, float boost, float threshold, string phrase, int debounceMs)
            {
                TokenIds = tokenIds;
                BoostScore = boost;
                Threshold = threshold;
                Phrase = phrase;
                DebounceMs = debounceMs;
            }
        }

        /// <summary>
        /// 把 options 中通过 KeywordsFile + Keywords 提供的关键词全部编码成 EncodedKeyword 列表。
        /// 任意一个关键词中包含 tokens.txt 中找不到的 token 时抛 <see cref="InvalidDataException"/>。
        /// </summary>
        public static List<EncodedKeyword> EncodeAll(SherpaOnnxKwsOptions options, SymbolTable symbols)
        {
            var encoded = new List<EncodedKeyword>();
            // token id 序列级去重: 不同来源 (KeywordsFile / 主关键词 / 变体 / 同音词) 可能产出
            // 完全相同的 token 序列, ContextGraph 中它们落在同一 end 节点, 保留首条即可。
            var seenTokenIds = new HashSet<string>(StringComparer.Ordinal);

            void AddUnique(in EncodedKeyword ek)
            {
                if (seenTokenIds.Add(string.Join(' ', ek.TokenIds))) encoded.Add(ek);
            }

            if (!string.IsNullOrWhiteSpace(options.KeywordsFile))
            {
                if (!File.Exists(options.KeywordsFile))
                    throw new FileNotFoundException(
                        $"Keywords file not found: {options.KeywordsFile}", options.KeywordsFile);
                foreach (var line in File.ReadAllLines(options.KeywordsFile))
                {
                    if (TryEncodeLine(line, symbols, options, out var ek)) AddUnique(ek);
                }
            }

            // 是否需要 AutoTokenizer (有任一关键词的 Tokens 为空)
            AutoTokenizer? auto = null;
            foreach (var kw in options.Keywords)
            {
                if (string.IsNullOrWhiteSpace(kw.Tokens))
                {
                    auto ??= BuildAutoTokenizer(options, symbols);
                    break;
                }
            }

            foreach (var kw in options.Keywords)
            {
                float boost = kw.BoostScore > 0 ? kw.BoostScore : 0f;
                float threshold = kw.Threshold > 0 ? kw.Threshold : 0f;
                int debounce = kw.DebounceMs > 0 ? kw.DebounceMs : options.DebounceMs;

                if (!string.IsNullOrWhiteSpace(kw.Tokens))
                {
                    int[] ids = EncodeTokens(kw.Tokens, symbols, kw.Keyword);
                    AddUnique(new EncodedKeyword(ids, boost, threshold, kw.Keyword, debounce));
                    continue;
                }

                if (auto == null)
                    throw new InvalidOperationException(
                        $"Keyword '{kw.Keyword}' has empty Tokens but AutoTokenizer was not initialized.");

                // 把 "主关键词原文 + 用户指定的发音变体" 一起送 AutoTokenizer,
                // 全部 token 序列共享同一 Phrase (= kw.Keyword)。
                // AutoTokenizer 内部还会按 3+3 上声变调再扩出额外子变体。
                var sources = new List<string>(1 + kw.Variants.Count) { kw.Keyword };
                foreach (var v in kw.Variants)
                {
                    if (!string.IsNullOrWhiteSpace(v)) sources.Add(v);
                }
                foreach (var src in sources)
                {
                    foreach (var variant in auto.TokenizeAll(src))
                    {
                        int[] ids = EncodeTokens(variant, symbols, kw.Keyword);
                        AddUnique(new EncodedKeyword(ids, boost, threshold, kw.Keyword, debounce));
                    }
                }
            }

            if (encoded.Count == 0)
                throw new InvalidOperationException(
                    "At least one keyword must be provided via KeywordsFile or AddKeyword().");
            return encoded;
        }

        private static AutoTokenizer BuildAutoTokenizer(SherpaOnnxKwsOptions options, SymbolTable symbols)
        {
            string cmuPath = options.CmuPhoneDictionaryPath;
            if (string.IsNullOrWhiteSpace(cmuPath))
            {
                // 默认查 tokens.txt 同目录的 en.phone (sherpa-onnx 官方模型自带)
                string? dir = Path.GetDirectoryName(options.TokensFilePath);
                if (!string.IsNullOrEmpty(dir))
                {
                    string candidate = Path.Combine(dir, "en.phone");
                    if (File.Exists(candidate)) cmuPath = candidate;
                }
            }

            CmuPhoneDictionary? cmu = null;
            if (!string.IsNullOrWhiteSpace(cmuPath) && File.Exists(cmuPath))
                cmu = new CmuPhoneDictionary(cmuPath);

            return new AutoTokenizer(symbols, cmu,
                expandPolyphone: options.ExpandPolyphoneVariants,
                maxVariants: options.MaxAutoVariantsPerKeyword,
                confusion: options.ConfusionVariants);
        }

        /// <summary>
        /// 解析关键词文件的一行: "tok1 tok2 ... [:score] [#threshold] [@phrase]"。
        /// 空行 / 无 token 行返回 false。
        /// </summary>
        private static bool TryEncodeLine(
            string line, SymbolTable symbols, SherpaOnnxKwsOptions opts, out EncodedKeyword encoded)
        {
            encoded = default;
            if (string.IsNullOrWhiteSpace(line)) return false;

            float score = 0f, threshold = 0f;
            string phrase = string.Empty;
            var tokenIds = new List<int>();

            foreach (var raw in line.Split(_whitespaceSeparators, StringSplitOptions.RemoveEmptyEntries))
            {
                if (raw.Length == 0) continue;
                if (symbols.Contains(raw))
                {
                    tokenIds.Add(symbols.GetId(raw));
                    continue;
                }
                switch (raw[0])
                {
                    case ':':
                        if (!float.TryParse(raw.AsSpan(1), System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture, out score))
                            throw new InvalidDataException(
                                $"Cannot parse boost score '{raw}' in line: {line}");
                        break;
                    case '#':
                        if (!float.TryParse(raw.AsSpan(1), System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture, out threshold))
                            throw new InvalidDataException(
                                $"Cannot parse threshold '{raw}' in line: {line}");
                        break;
                    case '@':
                        phrase = raw.Substring(1);
                        break;
                    default:
                        throw new InvalidDataException(
                            $"Token '{raw}' not in tokens.txt (line: {line}). 请检查关键词文件分词与 tokens.txt 是否匹配。");
                }
            }

            if (tokenIds.Count == 0) return false;
            if (string.IsNullOrEmpty(phrase))
            {
                // 当未给出 @phrase 时, 用 tokens 拼成的字符串作为关键词原文 (与 sherpa-onnx 行为一致)
                phrase = string.Concat(tokenIds.Select(symbols.GetSymbol));
            }
            encoded = new EncodedKeyword(
                tokenIds.ToArray(), score, threshold, phrase, opts.DebounceMs);
            return true;
        }

        private static int[] EncodeTokens(string tokensText, SymbolTable symbols, string keyword)
        {
            var ids = new List<int>();
            foreach (var raw in tokensText.Split(_whitespaceSeparators, StringSplitOptions.RemoveEmptyEntries))
            {
                if (!symbols.TryGetId(raw, out int id))
                    throw new InvalidDataException(
                        $"Token '{raw}' for keyword '{keyword}' is not in tokens.txt.");
                ids.Add(id);
            }
            if (ids.Count == 0)
                throw new ArgumentException($"Keyword '{keyword}' produced 0 tokens.", nameof(keyword));
            return ids.ToArray();
        }
    }
}
