using System.Text;
using System.Text.Json;

namespace Misaki.NET.Lang.English.Neural;

/// <summary>
/// Tokenizer for the DeepPhonemizer ForwardTransformer model. Mirrors the Python
/// SequenceTokenizer used during training (char_repeats expansion, lang start token,
/// end token, lowercase normalization).
/// </summary>
internal sealed class DeepPhonemizerTokenizer
{
    private readonly Dictionary<string, int> _textTokenToIdx;
    private readonly Dictionary<int, string> _phonemeIdxToToken;
    private readonly Dictionary<string, int> _langToStartIndex;
    private readonly int _charRepeats;
    private readonly bool _lowercase;
    private readonly bool _appendStartEnd;
    private readonly int _endIndex;
    private readonly int _phonemeEndIndex;
    private readonly HashSet<int> _phonemeSpecialIdx;

    public DeepPhonemizerTokenizer(string tokenizerJsonPath)
    {
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"DeepPhonemizer tokenizer config not found: {tokenizerJsonPath}");

        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var text = root.GetProperty("text_tokenizer");
        var phon = root.GetProperty("phoneme_tokenizer");

        _charRepeats = text.GetProperty("char_repeats").GetInt32();
        _lowercase = text.GetProperty("lowercase").GetBoolean();
        _appendStartEnd = text.GetProperty("append_start_end").GetBoolean();
        _endIndex = text.GetProperty("end_index").GetInt32();
        _phonemeEndIndex = phon.GetProperty("end_index").GetInt32();

        _textTokenToIdx = new(StringComparer.Ordinal);
        foreach (var prop in text.GetProperty("token_to_idx").EnumerateObject())
            _textTokenToIdx[prop.Name] = prop.Value.GetInt32();

        _langToStartIndex = new(StringComparer.Ordinal);
        foreach (var prop in text.GetProperty("lang_to_start_index").EnumerateObject())
            _langToStartIndex[prop.Name] = prop.Value.GetInt32();

        _phonemeIdxToToken = [];
        foreach (var prop in phon.GetProperty("idx_to_token").EnumerateObject())
        {
            if (int.TryParse(prop.Name, out var idx))
                _phonemeIdxToToken[idx] = prop.Value.GetString() ?? string.Empty;
        }

        _phonemeSpecialIdx =
        [
            phon.GetProperty("pad_index").GetInt32(),
            _phonemeEndIndex,
        ];
        foreach (var prop in phon.GetProperty("lang_to_start_index").EnumerateObject())
            _phonemeSpecialIdx.Add(prop.Value.GetInt32());
    }

    public bool HasLanguage(string lang) => _langToStartIndex.ContainsKey(lang);

    /// <summary>
    /// Encodes a word into model input ids. Replicates SequenceTokenizer.__call__:
    /// lowercase (if configured), each char repeated char_repeats times, with the
    /// language start token at the front and end token at the back (when append_start_end).
    /// </summary>
    public long[] Encode(string word, string lang)
    {
        if (!_langToStartIndex.TryGetValue(lang, out var langStart))
            throw new ArgumentException($"Unknown DeepPhonemizer language: {lang}");

        if (_lowercase) word = word.ToLowerInvariant();

        var ids = new List<long>(word.Length * _charRepeats + 2);
        if (_appendStartEnd) ids.Add(langStart);

        foreach (var c in word)
        {
            var key = c.ToString();
            if (!_textTokenToIdx.TryGetValue(key, out var idx))
                continue; // unseen char — skip; the model never trained on it
            for (int r = 0; r < _charRepeats; r++)
                ids.Add(idx);
        }

        if (_appendStartEnd) ids.Add(_endIndex);
        return [.. ids];
    }

    /// <summary>
    /// Decodes argmax indices into a phoneme string. Mirrors DeepPhonemizer's
    /// ForwardTransformer._generate: stops at the first phoneme end token, collapses
    /// adjacent duplicate ids (the model emits the same phoneme across each char_repeats
    /// expanded group), and drops pad/start/end specials.
    /// </summary>
    public string Decode(IReadOnlyList<int> indices)
    {
        var sb = new StringBuilder(indices.Count);
        int prev = -1;
        foreach (var idx in indices)
        {
            if (idx == _phonemeEndIndex) break;
            if (idx == prev) continue;
            prev = idx;
            if (_phonemeSpecialIdx.Contains(idx)) continue;
            if (_phonemeIdxToToken.TryGetValue(idx, out var tok))
                sb.Append(tok);
        }
        return sb.ToString();
    }
}
