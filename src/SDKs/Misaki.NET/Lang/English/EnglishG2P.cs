using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using Misaki.NET.Lang.English.Neural;
using Misaki.NET.Models;

namespace Misaki.NET.Lang.English;

/// <summary>
/// English Grapheme-to-Phoneme converter. Uses a pronunciation lexicon with morphological
/// stemming and number-to-words conversion; unknown words fall back to a neural model
/// (DeepPhonemizer ForwardTransformer, en_us_cmudict_ipa). Hybrid letter+digit tokens
/// (IPv6, MP3, H264, iPhone15) are pronounced segment-by-segment.
/// </summary>
public sealed class EnglishG2P : IG2P, IDisposable
{
    private const string DefaultModelDir = "runtimes/models/deepphonemizer";
    private const string ModelFileName = "deep_phonemizer.onnx";
    private const string TokenizerFileName = "deep_phonemizer_tokenizer.json";
    private const string ModelLanguage = "en_us";

    private readonly bool _british;
    private readonly string _unk;
    private readonly EnglishLexicon _lexicon;
    private readonly DeepPhonemizerSession _session;
    private readonly DeepPhonemizerTokenizer _tokenizer;
    private readonly ConcurrentDictionary<string, string> _neuralCache = new(StringComparer.Ordinal);
    private int _disposed;

    private static readonly HashSet<char> Vowels = new("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ");
    private static readonly HashSet<char> Consonants = new("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ");
    private static readonly HashSet<char> NonQuotePuncts = new(";:,.!?—…");
    private static readonly HashSet<char> Puncts = new(";:,.!?—…“”„");

    // The first two alternatives keep letter+digit hybrids (IPv6, MP3, H264, 5G, iPhone15)
    // as a single token so TryProcessHybridToken below can spell/pronounce each segment.
    private static readonly Regex SubtokenRegex = new(
        @"[A-Za-z]+\d+[A-Za-z\d]*|\d+[A-Za-z]+[A-Za-z\d]*|[A-Z]?[a-z]+(?:'[a-z]+)?|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z](?=[a-z])|[0-9]+(?:[,\.][0-9]+)*|\S",
        RegexOptions.Compiled);

    private static readonly Regex LinkRegex = new(
        @"\[([^\]]+)\]\(([^\)]*)\)",
        RegexOptions.Compiled);

    /// <summary>
    /// Creates a new English G2P instance. The DeepPhonemizer model files are loaded from
    /// <paramref name="modelDir"/> (defaults to runtimes/models/deepphonemizer relative to
    /// the executing assembly).
    /// </summary>
    /// <param name="british">Use British English pronunciation.</param>
    /// <param name="unk">Placeholder phoneme for tokens that fail every pathway.</param>
    /// <param name="modelDir">Directory containing deep_phonemizer.onnx and the tokenizer JSON.</param>
    public EnglishG2P(bool british = false, string unk = "❓", string? modelDir = null)
    {
        _british = british;
        _unk = unk;
        _lexicon = new EnglishLexicon(british);

        var dir = ResolveModelDir(modelDir);
        _session = new DeepPhonemizerSession(Path.Combine(dir, ModelFileName));
        _tokenizer = new DeepPhonemizerTokenizer(Path.Combine(dir, TokenizerFileName));

        if (!_tokenizer.HasLanguage(ModelLanguage))
            throw new InvalidOperationException(
                $"DeepPhonemizer tokenizer does not expose language '{ModelLanguage}'. " +
                "Re-export with an en_us checkpoint.");
    }

    private static string ResolveModelDir(string? modelDir)
    {
        if (!string.IsNullOrWhiteSpace(modelDir) && Directory.Exists(modelDir))
            return modelDir!;

        // Search next to the executing assembly first (publish/output layout), then
        // walk up to the repo root (helps tests run from bin/Debug/.../net10.0).
        var baseDir = AppContext.BaseDirectory;
        var candidates = new List<string> { Path.Combine(baseDir, DefaultModelDir) };
        var probe = new DirectoryInfo(baseDir);
        for (int i = 0; i < 6 && probe is not null; i++, probe = probe.Parent)
            candidates.Add(Path.Combine(probe.FullName, DefaultModelDir));

        foreach (var c in candidates)
            if (Directory.Exists(c) && File.Exists(Path.Combine(c, ModelFileName)))
                return c;

        throw new DirectoryNotFoundException(
            $"DeepPhonemizer model dir not found. Tried: {string.Join(", ", candidates)}");
    }

    /// <inheritdoc />
    public G2PResult Process(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new G2PResult(string.Empty);

        text = text.Trim();

        var (cleanText, features) = Preprocess(text);
        var tokens = Tokenize(cleanText, features);
        AssignAliases(tokens);
        RetagArticleA(tokens);

        var ctx = new TokenContext();
        for (int i = tokens.Count - 1; i >= 0; i--)
        {
            var tk = tokens[i];
            if (tk.Phonemes != null) continue;

            if (tk.Text.Length == 1 && Puncts.Contains(tk.Text[0]))
            {
                tk.Phonemes = tk.Text;
                tk.Properties.Rating = 4;
                ctx = UpdateContext(ctx, tk.Phonemes, tk);
                continue;
            }

            var lookupText = tk.Properties.Alias ?? tk.Text;
            var word = NormalizeWord(lookupText);
            double? stress = GetStress(lookupText);

            var (ps, rating) = _lexicon.GetWord(word, tk.Tag, stress, ctx);
            if (ps != null)
            {
                tk.Phonemes = EnglishLexicon.ApplyStress(ps, tk.Properties.Stress);
                tk.Properties.Rating = rating;
                ctx = UpdateContext(ctx, tk.Phonemes, tk);
                continue;
            }

            if (EnglishLexicon.IsNumber(word, tk.Properties.IsHead))
            {
                var (numPs, numRating) = _lexicon.GetNumber(word, tk.Properties.Currency, tk.Properties.IsHead, tk.Properties.NumFlags);
                if (numPs != null)
                {
                    tk.Phonemes = EnglishLexicon.ApplyStress(numPs, tk.Properties.Stress);
                    tk.Properties.Rating = numRating;
                    ctx = UpdateContext(ctx, tk.Phonemes, tk);
                    continue;
                }
            }

            var hybridPs = TryProcessHybridToken(word, tk.Tag, ctx);
            if (hybridPs != null)
            {
                tk.Phonemes = EnglishLexicon.ApplyStress(hybridPs, tk.Properties.Stress);
                tk.Properties.Rating = 3;
                ctx = UpdateContext(ctx, tk.Phonemes, tk);
                continue;
            }

            // Neural fallback. Strip non-letter chars before invoking the model — DP
            // was only trained on a-z (+ a few accented German chars we don't use).
            var letters = ExtractLetters(word);
            if (letters.Length > 0)
            {
                var neural = PredictNeural(letters);
                if (!string.IsNullOrEmpty(neural))
                {
                    tk.Phonemes = EnglishLexicon.ApplyStress(neural, tk.Properties.Stress);
                    tk.Properties.Rating = 2;
                    ctx = UpdateContext(ctx, tk.Phonemes, tk);
                    continue;
                }
            }

            tk.Phonemes = _unk;
            ctx = UpdateContext(ctx, tk.Phonemes, tk);
        }

        foreach (var tk in tokens)
        {
            if (!string.IsNullOrEmpty(tk.Phonemes))
                tk.Phonemes = tk.Phonemes.Replace('ɾ', 'T').Replace('ʔ', 't');
        }

        var result = string.Concat(tokens.Select(tk => (tk.Phonemes ?? _unk) + tk.Whitespace));
        return new G2PResult(result, tokens);
    }

    private string PredictNeural(string letters)
    {
        var key = letters.ToLowerInvariant();
        if (_neuralCache.TryGetValue(key, out var cached))
            return cached;

        var ids = _tokenizer.Encode(key, ModelLanguage);
        if (ids.Length == 0) return string.Empty;

        var argmax = _session.Predict(ids);
        var rawIpa = _tokenizer.Decode(argmax);
        var mapped = IpaToMisakiMapper.Map(rawIpa, _british);

        _neuralCache.TryAdd(key, mapped);
        return mapped;
    }

    private static string ExtractLetters(string word)
    {
        bool allLetters = true;
        for (int i = 0; i < word.Length; i++)
        {
            if (!char.IsLetter(word[i])) { allLetters = false; break; }
        }
        if (allLetters) return word;

        var buf = new char[word.Length];
        int n = 0;
        foreach (var c in word)
            if (char.IsLetter(c)) buf[n++] = c;
        return n == 0 ? string.Empty : new string(buf, 0, n);
    }

    private static (string text, Dictionary<int, object> features) Preprocess(string text)
    {
        var features = new Dictionary<int, object>();
        var result = new System.Text.StringBuilder();
        int tokenIndex = 0;
        int lastEnd = 0;

        foreach (Match m in LinkRegex.Matches(text))
        {
            var before = text[lastEnd..m.Index];
            result.Append(before);
            tokenIndex += before.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

            var f = m.Groups[2].Value;
            object? feature = null;

            if (int.TryParse(f.TrimStart('+', '-'), out _) && (f[0] == '+' || f[0] == '-' || char.IsDigit(f[0])))
                feature = int.Parse(f);
            else if (f is "0.5" or "+0.5")
                feature = 0.5;
            else if (f == "-0.5")
                feature = -0.5;
            else if (f.Length > 1 && f[0] == '/' && f[^1] == '/')
                feature = f[..^1];
            else if (f.Length > 1 && f[0] == '#' && f[^1] == '#')
                feature = "#" + f[1..].TrimEnd('#');

            if (feature != null)
                features[tokenIndex] = feature;

            result.Append(m.Groups[1].Value);
            tokenIndex++;
            lastEnd = m.Index + m.Length;
        }

        if (lastEnd < text.Length)
            result.Append(text[lastEnd..]);

        return (result.ToString(), features);
    }

    private static List<MToken> Tokenize(string text, Dictionary<int, object> features)
    {
        var tokens = new List<MToken>();
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        for (int i = 0; i < words.Length; i++)
        {
            var word = words[i];
            var subtokens = SubtokenRegex.Matches(word);

            foreach (Match st in subtokens)
            {
                var tk = new MToken
                {
                    Text = st.Value,
                    Tag = GuessPOS(st.Value),
                    Whitespace = "",
                    Properties = new TokenProperties
                    {
                        IsHead = true,
                        NumFlags = "",
                        PreSpace = false,
                    }
                };

                if (features.TryGetValue(i, out var feat))
                {
                    if (feat is int stressInt)
                        tk.Properties.Stress = stressInt;
                    else if (feat is double stressDouble)
                        tk.Properties.Stress = stressDouble;
                    else if (feat is string fs && fs.StartsWith('/'))
                    {
                        tk.Phonemes = fs[1..];
                        tk.Properties.Rating = 5;
                    }
                    else if (feat is string nf && nf.StartsWith('#'))
                    {
                        tk.Properties.NumFlags = nf[1..];
                    }
                }

                tokens.Add(tk);
            }

            if (tokens.Count > 0 && i < words.Length - 1)
                tokens[^1].Whitespace = " ";
        }

        return tokens;
    }

    private static void AssignAliases(List<MToken> tokens)
    {
        for (int j = 1; j < tokens.Count - 1; j++)
        {
            var tk = tokens[j];
            if (tk.Properties.Alias != null || tk.Phonemes != null)
                continue;
            if (tk.Text != "2")
                continue;

            var prevText = tokens[j - 1].Text;
            var nextText = tokens[j + 1].Text;
            if (prevText.Length == 0 || nextText.Length == 0)
                continue;

            if (char.IsLetter(prevText[^1]) && char.IsLetter(nextText[0]))
                tk.Properties.Alias = "to";
        }
    }

    private static void RetagArticleA(List<MToken> tokens)
    {
        for (int j = 0; j < tokens.Count - 1; j++)
        {
            var tk = tokens[j];
            if (tk.Phonemes != null) continue;
            if (tk.Text is not ("a" or "A")) continue;

            var next = tokens[j + 1];
            if (next.Text.Length == 0) continue;
            if (!char.IsLetter(next.Text[0])) continue;

            tk.Tag = "DT";
        }
    }

    private static string GuessPOS(string word)
    {
        if (word.All(char.IsAsciiDigit)) return "CD";
        if (word.Length == 1 && !char.IsLetterOrDigit(word[0])) return ".";
        if (word is "$" or "£" or "€") return "$";
        if (word == word.ToUpperInvariant() && word.Length > 1 && word.All(char.IsLetter)) return "NNP";
        return "NN";
    }

    private string? TryProcessHybridToken(string word, string? tag, TokenContext ctx)
    {
        if (word.Length < 2) return null;
        bool hasDigit = false, hasLetter = false;
        foreach (var c in word)
        {
            if (char.IsAsciiDigit(c)) hasDigit = true;
            else if (char.IsLetter(c)) hasLetter = true;
            else return null;
        }
        if (!hasDigit || !hasLetter) return null;

        var parts = new List<string>();
        bool prevSegmentSpelled = false;
        int i = 0;
        while (i < word.Length)
        {
            int j = i;
            bool isDigit = char.IsAsciiDigit(word[i]);
            while (j < word.Length && char.IsAsciiDigit(word[j]) == isDigit) j++;
            var segment = word[i..j];

            if (isDigit)
            {
                if (prevSegmentSpelled)
                {
                    foreach (var d in segment)
                    {
                        var (digitPs, _) = _lexicon.GetNumber(d.ToString(), null, true, "");
                        if (digitPs == null) return null;
                        parts.Add(digitPs);
                    }
                }
                else
                {
                    var (numPs, _) = _lexicon.GetNumber(segment, null, true, "");
                    if (numPs == null) return null;
                    parts.Add(numPs);
                }
                prevSegmentSpelled = false;
            }
            else
            {
                var letterPs = ResolveLetterSegment(segment, tag, ctx);
                if (letterPs == null) return null;
                parts.Add(letterPs);
                prevSegmentSpelled = segment.All(char.IsUpper);
            }

            i = j;
        }

        return string.Concat(parts);
    }

    private string? ResolveLetterSegment(string segment, string? tag, TokenContext ctx)
    {
        var (wordPs, _) = _lexicon.GetWord(segment, tag, null, ctx);
        if (wordPs != null) return wordPs;

        if (TrySplitCamelCase(segment, out var pieces))
        {
            var buf = new System.Text.StringBuilder();
            foreach (var piece in pieces)
            {
                var (pPs, _) = _lexicon.GetWord(piece, tag, null, ctx);
                if (pPs == null) return SpellSegment(segment, ctx);
                buf.Append(pPs);
            }
            return buf.ToString();
        }

        return SpellSegment(segment, ctx);
    }

    private string? SpellSegment(string segment, TokenContext ctx)
    {
        var upper = segment.ToUpperInvariant();
        var (spelledPs, _) = _lexicon.Lookup(upper, "NNP", null, ctx);
        return string.IsNullOrEmpty(spelledPs) ? null : spelledPs;
    }

    private static bool TrySplitCamelCase(string segment, out List<string> pieces)
    {
        pieces = [];
        int start = 0;
        for (int k = 1; k < segment.Length; k++)
        {
            if (char.IsLower(segment[k - 1]) && char.IsUpper(segment[k]))
            {
                pieces.Add(segment[start..k]);
                start = k;
            }
        }
        if (pieces.Count == 0) return false;
        pieces.Add(segment[start..]);
        return true;
    }

    private static string NormalizeWord(string text)
    {
        text = text.Replace('‘', '\'').Replace('’', '\'');
        text = text.Normalize(System.Text.NormalizationForm.FormKC);
        return text;
    }

    private static double? GetStress(string word)
    {
        if (word == word.ToLowerInvariant()) return null;
        return word == word.ToUpperInvariant() ? 2.0 : 0.5;
    }

    private static TokenContext UpdateContext(TokenContext ctx, string? phonemes, MToken token)
    {
        bool? futureVowel = ctx.FutureVowel;
        if (!string.IsNullOrEmpty(phonemes))
        {
            foreach (var c in phonemes)
            {
                if (Vowels.Contains(c)) { futureVowel = true; break; }
                if (Consonants.Contains(c)) { futureVowel = false; break; }
                if (NonQuotePuncts.Contains(c)) { futureVowel = null; break; }
            }
        }

        bool futureTo = token.Text is "to" or "To" || (token.Text == "TO" && token.Tag is "TO" or "IN");
        return new TokenContext { FutureVowel = futureVowel, FutureTo = futureTo };
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _session.Dispose();
    }
}
