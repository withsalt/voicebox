using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace VoiceBox.TTS.Hojo.Internal;

/// <summary>Hojo character BPE, including serialized regexes and added-token normalization order.</summary>
internal sealed class HojoTokenizer
{
    private readonly Dictionary<string, int> _vocabulary = new(StringComparer.Ordinal);
    private readonly Dictionary<(string, string), int> _ranks = [];
    private readonly TokenTrie _raw = new();
    private readonly TokenTrie _normalized = new();
    private readonly Regex _replace;
    private readonly Regex _split;
    private readonly string _replacement;
    private readonly bool[] _audioIds;
    public int SpeechEndId { get; }

    public HojoTokenizer(string path, int vocabularySize = 17685)
    {
        _audioIds = new bool[vocabularySize];
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE" || model.GetProperty("byte_fallback").GetBoolean()
            || model.GetProperty("fuse_unk").GetBoolean())
            throw new InvalidDataException("Expected Hojo character BPE without byte fallback or fused unknowns.");
        foreach (var token in model.GetProperty("vocab").EnumerateObject())
        {
            int id = token.Value.GetInt32();
            if (id < 0 || id >= vocabularySize) throw new InvalidDataException("Token id exceeds the model vocabulary.");
            _vocabulary.Add(token.Name, id);
        }
        int rank = 0;
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
            _ranks.Add((merge[0].GetString()!, merge[1].GetString()!), rank++);
        foreach (var added in root.GetProperty("added_tokens").EnumerateArray())
        {
            string content = added.GetProperty("content").GetString()!;
            int id = added.GetProperty("id").GetInt32();
            if (id < 0 || id >= _audioIds.Length || added.GetProperty("single_word").GetBoolean()
                || added.GetProperty("lstrip").GetBoolean() || added.GetProperty("rstrip").GetBoolean())
                throw new InvalidDataException("Unsupported added-token semantics.");
            _vocabulary[content] = id;
            (added.GetProperty("normalized").GetBoolean() ? _normalized : _raw).Add(content, id);
        }
        foreach (var (name, id) in _vocabulary)
            if (name.Length > 2 && name[0] == '[' && name[^1] == ']' && int.TryParse(name.AsSpan(1, name.Length - 2), NumberStyles.None, CultureInfo.InvariantCulture, out _))
                _audioIds[id] = true;
        var normalizers = root.GetProperty("normalizer").GetProperty("normalizers");
        if (normalizers[0].GetProperty("type").GetString() != "Lowercase") throw new InvalidDataException("Unexpected tokenizer normalization.");
        _replace = new Regex(normalizers[1].GetProperty("pattern").GetProperty("Regex").GetString()!, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));
        _replacement = normalizers[1].GetProperty("content").GetString()!;
        _split = new Regex(root.GetProperty("pre_tokenizer").GetProperty("pretokenizers")[1].GetProperty("pattern").GetProperty("Regex").GetString()!, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(1));
        SpeechEndId = Id("[target_speech_end]");
    }

    public bool IsAudio(int id) => _audioIds[id];

    public static void ValidateText(string text)
    {
        if (text.Contains("[target_", StringComparison.OrdinalIgnoreCase) || text.Contains("[spk_", StringComparison.OrdinalIgnoreCase)
            || text.Contains("[ref_", StringComparison.OrdinalIgnoreCase) || text.Contains("[speech_", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Text must not contain Hojo prompt control tokens.", nameof(text));
    }

    public int[] EncodePrompt(string text, HojoVoicePrompt? reference = null)
    {
        ValidateText(text);
        if (reference is not null)
        {
            var prompt = new StringBuilder().Append("[ref_text_start]").Append(reference.Text).Append("[ref_text_end] ")
                .Append("[target_text_start]").Append(text).Append("[target_text_end][ref_speech_start]");
            foreach (int code in reference.Codes) prompt.Append('[').Append(code.ToString(CultureInfo.InvariantCulture)).Append(']');
            prompt.Append("[ref_speech_end][target_speech_start]");
            var encoded = new List<int>();
            EncodeAdded(prompt.ToString(), _raw, encoded, normalize: true);
            return [.. encoded];
        }
        var ids = new List<int>(text.Length + 21) { Id("[target_text_start]") };
        EncodeAdded(text, _raw, ids, normalize: true);
        ids.Add(Id("[target_text_end]"));
        ids.Add(Id("[spk_start]"));
        for (int i = 0; i < 16; i++) ids.Add(Id($"[spk_emb_{i}]"));
        ids.Add(Id("[spk_end]"));
        ids.Add(Id("[target_speech_start]"));
        return [.. ids];
    }

    private int Id(string token) => _vocabulary.TryGetValue(token, out int id) ? id : throw new InvalidDataException($"Missing token {token}.");

    private void EncodeAdded(string text, TokenTrie trie, List<int> output, bool normalize)
    {
        int plainStart = 0, i = 0;
        while (i < text.Length)
        {
            if (trie.Match(text, i) is { } token)
            {
                EncodePlain(text[plainStart..i], output, normalize);
                output.Add(token.Id);
                i += token.Length;
                plainStart = i;
            }
            else i++;
        }
        EncodePlain(text[plainStart..], output, normalize);
    }

    private void EncodePlain(string text, List<int> output, bool normalize)
    {
        if (text.Length == 0) return;
        if (normalize)
        {
            // Rust/tokenizers uses full Unicode lowercase: U+0130 expands to i + combining dot.
            string lowered = text.Replace("\u0130", "i\u0307", StringComparison.Ordinal).ToLowerInvariant();
            EncodeAdded(_replace.Replace(lowered, _replacement), _normalized, output, normalize: false);
            return;
        }
        int start = 0, offset = 0;
        foreach (Rune rune in text.EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) == UnicodeCategory.DecimalDigitNumber)
            {
                EncodeSplit(text[start..offset], output);
                EncodeBpe(rune.ToString(), output);
                start = offset + rune.Utf16SequenceLength;
            }
            offset += rune.Utf16SequenceLength;
        }
        EncodeSplit(text[start..], output);
    }

    private void EncodeSplit(string text, List<int> output)
    {
        int start = 0;
        foreach (Match match in _split.Matches(text))
        {
            EncodeBpe(text[start..match.Index], output);
            EncodeBpe(match.Value, output);
            start = match.Index + match.Length;
        }
        EncodeBpe(text[start..], output);
    }

    private void EncodeBpe(string text, List<int> output)
    {
        var pieces = new List<string>(text.Length);
        foreach (Rune rune in text.EnumerateRunes()) pieces.Add(rune.ToString());
        while (pieces.Count > 1)
        {
            int best = int.MaxValue, index = -1;
            for (int i = 0; i + 1 < pieces.Count; i++)
                if (_ranks.TryGetValue((pieces[i], pieces[i + 1]), out int rank) && rank < best) { best = rank; index = i; }
            if (index < 0) break;
            pieces[index] += pieces[index + 1];
            pieces.RemoveAt(index + 1);
        }
        foreach (string piece in pieces) output.Add(_vocabulary.GetValueOrDefault(piece, 0));
    }

    private sealed class TokenTrie
    {
        private readonly Dictionary<char, TokenTrie> _children = [];
        private int? _id;
        public void Add(string text, int id)
        {
            var node = this;
            foreach (char c in text)
            {
                if (!node._children.TryGetValue(c, out var child)) node._children.Add(c, child = new());
                node = child;
            }
            node._id = id;
        }
        public (int Id, int Length)? Match(string text, int start)
        {
            var node = this;
            (int, int)? result = null;
            for (int i = start; i < text.Length && node._children.TryGetValue(text[i], out node); i++)
                if (node._id is { } id) result = (id, i - start + 1);
            return result;
        }
    }
}
