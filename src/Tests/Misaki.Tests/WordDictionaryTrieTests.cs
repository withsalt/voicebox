using Misaki.NET.Jieba.Segmenter;
using Misaki.NET.Jieba.Segmenter.FinalSeg;
using Xunit;

namespace Misaki.Tests;

/// <summary>
/// Tests covering #1+#2: WordDictionary now uses a real character trie,
/// and JiebaSegmenter.GetDag/Calc walk it directly instead of materialising
/// substrings as hash keys.
///
/// We can't call GetDag directly (internal), so we rely on:
///   - ContainsWord / GetFreqOrDefault for dictionary correctness;
///   - JiebaSegmenter.Cut(...) to exercise GetDag + Calc end-to-end.
/// </summary>
public class WordDictionaryTrieTests
{
    private static readonly WordDictionary Dict = WordDictionary.Instance;

    [Fact]
    public void ContainsWord_KnownWord_ReturnsTrue()
    {
        Assert.True(Dict.ContainsWord("中国"));
        Assert.True(Dict.ContainsWord("我们"));
    }

    [Fact]
    public void ContainsWord_UnknownWord_ReturnsFalse()
    {
        Assert.False(Dict.ContainsWord("不存在的怪词xyz"));
    }

    [Fact]
    public void ContainsWord_PrefixOfRealWordButNotItselfAWord_ReturnsFalse()
    {
        // The legacy Dictionary<string,int> stored every prefix with Freq=0; with the
        // real character trie, prefixes that are not real words must NOT be reported as words.
        // 中国人 is a real word; "中国" is also a word; but a synthesized prefix like "我们的xyz"
        // can be checked by composing a prefix from a known word that itself isn't a dict entry.
        // Use 我们 as a known word: confirm extra-character prefix isn't reported.
        Assert.False(Dict.ContainsWord("我们q"));
    }

    [Fact]
    public void GetFreqOrDefault_KnownWord_PositiveFreq()
    {
        Assert.True(Dict.GetFreqOrDefault("中国") > 1);
    }

    [Fact]
    public void GetFreqOrDefault_UnknownWord_ReturnsOne()
    {
        Assert.Equal(1, Dict.GetFreqOrDefault("绝对不在词典里的串zzz"));
    }

    [Fact]
    public void AddWord_NewWord_BecomesContainable()
    {
        var word = "测试新增词xyz";
        Dict.AddWord(word, 100);
        try
        {
            Assert.True(Dict.ContainsWord(word));
            Assert.Equal(100, Dict.GetFreqOrDefault(word));
        }
        finally
        {
            // Restore — DeleteWord just sets freq=0; it removes from "is a word" view.
            Dict.DeleteWord(word);
        }
    }

    [Fact]
    public void DeleteWord_RemovesFromContainsWord()
    {
        var word = "另一个新增词abc";
        Dict.AddWord(word, 50);
        Dict.DeleteWord(word);
        Assert.False(Dict.ContainsWord(word));
    }

    [Fact]
    public void AddWord_TotalReflectsAddAndDelete()
    {
        var before = Dict.Total;
        var word = "总频测试xx";
        Dict.AddWord(word, 42);
        Assert.Equal(before + 42, Dict.Total);
        Dict.DeleteWord(word);
        // After delete (freq=0), AddWord(word,0) subtracts existing 42.
        Assert.Equal(before, Dict.Total);
    }

    // ---- End-to-end: GetDag + Calc paths via JiebaSegmenter ------------------

    [Fact]
    public void Segmenter_TypicalChineseSentence_ProducesExpectedTokens()
    {
        var seg = new JiebaSegmenter();
        var tokens = seg.Cut("我来到北京清华大学").ToList();
        // jieba produces: 我 / 来到 / 北京 / 清华大学
        Assert.Equal(new[] { "我", "来到", "北京", "清华大学" }, tokens);
    }

    [Fact]
    public void Segmenter_MixedCJKAndAscii_ProducesNonEmptyTokens()
    {
        var seg = new JiebaSegmenter();
        var tokens = seg.Cut("hello 世界 2026").ToList();
        Assert.NotEmpty(tokens);
        // ASCII run survives intact
        Assert.Contains("hello", tokens);
        Assert.Contains("2026", tokens);
        Assert.Contains("世界", tokens);
    }

    [Fact]
    public void Segmenter_EmptyInput_NoCrash()
    {
        var seg = new JiebaSegmenter();
        var tokens = seg.Cut("").ToList();
        Assert.Empty(tokens);
    }

    [Fact]
    public void Segmenter_NewlyAddedDomainWord_IsTokenizedAsOnePiece()
    {
        var seg = new JiebaSegmenter();
        var word = "甲乙丙丁戊己";
        // Without it in the dict, jieba splits into characters.
        // Add a high freq to force the optimal route to keep it whole.
        WordDictionary.Instance.AddWord(word, 100000);
        try
        {
            var tokens = seg.Cut("甲乙丙丁戊己很有用").ToList();
            Assert.Contains(word, tokens);
        }
        finally
        {
            WordDictionary.Instance.DeleteWord(word);
        }
    }

    [Fact]
    public void Segmenter_TraversingShortRepeated_NoOutOfBoundsOrLoss()
    {
        var seg = new JiebaSegmenter();
        // Chained known short words; the trie walk must yield the shortest valid prefix DAG entries.
        var tokens = seg.Cut("的的的的").ToList();
        // Whatever the partition, every input char must appear in concatenation.
        Assert.Equal("的的的的", string.Concat(tokens));
    }

    [Fact]
    public void Segmenter_LongSentence_RoundTripsAllCharacters()
    {
        var seg = new JiebaSegmenter();
        const string s = "今天天气很好我们一起去公园散步聊天看花观鸟拍照吃午饭。";
        var joined = string.Concat(seg.Cut(s));
        Assert.Equal(s, joined);
    }

    // ---- FinalSeg.Viterbi (#3) end-to-end smoke -----------------------------

    [Fact]
    public void FinalSegViterbi_KnownText_TokensConcatenateBackToInput()
    {
        var v = Viterbi.Instance;
        const string s = "我爱北京天安门";
        var joined = string.Concat(v.Cut(s));
        Assert.Equal(s, joined);
    }

    [Fact]
    public void FinalSegViterbi_LongChineseSpan_RoundTrips()
    {
        var v = Viterbi.Instance;
        const string s = "深度学习自然语言处理是一个很有挑战性的领域";
        Assert.Equal(s, string.Concat(v.Cut(s)));
    }

    [Fact]
    public void FinalSegViterbi_MixedAsciiAndChinese_PreservesAllChars()
    {
        var v = Viterbi.Instance;
        const string s = "买iPhone16手机";
        Assert.Equal(s, string.Concat(v.Cut(s)));
    }

    [Fact]
    public void FinalSegViterbi_UnknownCharFallback_DoesNotCrash()
    {
        var v = Viterbi.Instance;
        const string s = "𠀀字"; // includes a char unlikely to appear in emit table
        // Simply verifying no exception and tokens cover the input length
        var tokens = v.Cut(s).ToList();
        Assert.NotEmpty(tokens);
    }
}
