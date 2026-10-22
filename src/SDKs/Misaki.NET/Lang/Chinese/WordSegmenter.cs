using Misaki.NET.Jieba.Segmenter;
using Misaki.NET.Jieba.Segmenter.PosSeg;

namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Chinese word segmenter backed by jieba.NET.
/// Provides word segmentation with POS tagging using jieba's statistical model (HMM + Viterbi).
/// Thread-safe: each instance holds its own PosSegmenter.
/// </summary>
public sealed class WordSegmenter
{
    private readonly PosSegmenter _posSegmenter;
    private readonly JiebaSegmenter _jiebaSegmenter;

    private static readonly HashSet<char> PunctuationChars = new(";:,.!?—…\"()\u201C\u201D '/");

    /// <summary>Creates a new word segmenter backed by jieba.NET.</summary>
    /// <param name="lookup">Pinyin lookup (retained for API compatibility but not used for segmentation).</param>
    public WordSegmenter(PinyinLookup lookup)
    {
        _posSegmenter = new PosSegmenter();
        _jiebaSegmenter = new JiebaSegmenter();
    }

    /// <summary>
    /// Segments Chinese text into words with POS tags using jieba.NET.
    /// Returns list of (word, pos) tuples compatible with the downstream pipeline.
    /// </summary>
    public List<(string Word, string Pos)> Segment(string text)
    {
        if (string.IsNullOrEmpty(text))
            return new List<(string, string)>();

        var result = new List<(string, string)>();
        var pairs = _posSegmenter.Cut(text);

        foreach (var pair in pairs)
        {
            if (string.IsNullOrEmpty(pair.Word))
                continue;

            // Map jieba's POS flags to the format expected by the downstream pipeline
            var word = pair.Word;
            var pos = MapPos(word, pair.Flag);
            result.Add((word, pos));
        }

        return result;
    }

    /// <summary>
    /// Segments text into words only (no POS tags).
    /// </summary>
    public List<string> SegmentWords(string text)
    {
        if (string.IsNullOrEmpty(text))
            return new List<string>();

        return _jiebaSegmenter.Cut(text, hmm: true).ToList();
    }

    /// <summary>
    /// Segments a word using jieba's search mode (cut_for_search).
    /// Used by ToneSandhi for sub-word splitting in neural tone analysis.
    /// Mirrors Python tone_sandhi._split_word's use of jieba.cut_for_search.
    /// </summary>
    public IEnumerable<string> CutForSearch(string text)
    {
        if (string.IsNullOrEmpty(text))
            return Array.Empty<string>();

        return _jiebaSegmenter.CutForSearch(text, hmm: true);
    }

    /// <summary>
    /// Forces jieba to recognize <paramref name="word"/> as a single token.
    /// Used by ChineseG2P when registering user phrases so polyphone overrides
    /// can match against a stable segmentation.
    /// </summary>
    public void AddWord(string word, string? pos = null)
    {
        if (string.IsNullOrWhiteSpace(word))
            return;
        _jiebaSegmenter.AddWord(word, freq: 0, tag: pos);
    }

    /// <summary>
    /// Maps jieba.NET POS tags to the format expected by the Misaki pipeline.
    /// jieba uses ICTCLAS-style tags (n, v, a, d, m, q, r, p, c, u, x, eng, etc.)
    /// which are already compatible with what ToneSandhi expects.
    /// </summary>
    private static string MapPos(string word, string flag)
    {
        // Whitespace
        if (string.IsNullOrWhiteSpace(word))
            return "x";

        // Punctuation detection - jieba may tag punctuation as 'x' or miss it
        if (word.Length == 1 && PunctuationChars.Contains(word[0]))
            return "x";

        // jieba already uses standard ICTCLAS tags that ToneSandhi expects:
        // n=noun, v=verb, a=adjective, d=adverb, m=numeral, q=quantifier,
        // r=pronoun, p=preposition, c=conjunction, u=auxiliary,
        // x=other, eng=English, nr=person name, ns=place name, etc.
        if (string.IsNullOrEmpty(flag))
            return "n";

        return flag;
    }
}
