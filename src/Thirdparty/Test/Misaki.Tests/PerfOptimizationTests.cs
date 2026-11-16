using Xunit;
using Misaki.NET.Lang.English;
using Misaki.NET.Lang.Chinese;

namespace Misaki.Tests;

public class PerfOptimizationTests
{
    [Fact]
    public void Perf3_ModifiedTone_TwoToneThree_FirstBecomesTwo()
    {
        // 你好 → ni3 + hao3 → ni2 + hao3 (tone-3 sandhi)
        var finals = new List<string> { "i3", "ao3" };
        var result = ToneSandhi.ModifiedTone("你好", "l", finals);
        Assert.Equal(2, result.Count);
        // First tone-3 becomes tone-2
        Assert.Equal('2', result[0][^1]);
        Assert.Equal('3', result[1][^1]);
    }

    [Fact]
    public void Perf3_ModifiedTone_BuBeforeTone4_BecomesTone2()
    {
        // 不是 → bu4 + shi4 → bu2 + shi4
        var finals = new List<string> { "u4", "i4" };
        var result = ToneSandhi.ModifiedTone("不是", "d", finals);
        // 不's tone should become 2 before a tone-4 syllable
        Assert.Equal('2', result[0][^1]);
        Assert.Equal('4', result[1][^1]);
    }

    [Fact]
    public void Perf3_ModifiedTone_YiBeforeTone4_BecomesTone2()
    {
        // 一样 → yi1 + yang4 → yi2 + yang4
        var finals = new List<string> { "i1", "ang4" };
        var result = ToneSandhi.ModifiedTone("一样", "d", finals);
        Assert.Equal('2', result[0][^1]);
    }

    [Fact]
    public void Perf3_ModifiedTone_DiYi_StaysYi1()
    {
        // 第一 keeps yi1
        var finals = new List<string> { "i4", "i1" };
        var result = ToneSandhi.ModifiedTone("第一", "m", finals);
        // 一 stays tone 1
        Assert.Equal('1', result[1][^1]);
    }

    [Fact]
    public void Perf3_ModifiedTone_VBuV_BecomesNeutral()
    {
        // 去不去 → qu4 + bu4 + qu4 → 不 becomes neutral tone (5)
        var finals = new List<string> { "u4", "u4", "u4" };
        var result = ToneSandhi.ModifiedTone("去不去", "v", finals);
        Assert.Equal('5', result[1][^1]);
    }

    [Fact]
    public void Perf3_ModifiedTone_EmptyFinals_NoCrash()
    {
        var result = ToneSandhi.ModifiedTone("", "", new List<string>());
        Assert.Empty(result);
    }

    [Fact]
    public void Perf3_ModifiedTone_DoesNotMutateInput()
    {
        // The method should return a new list without modifying the input
        var input = new List<string> { "i3", "ao3" };
        var snapshot = new List<string>(input);
        ToneSandhi.ModifiedTone("你好", "l", input);
        Assert.Equal(snapshot, input);
    }

    [Fact]
    public void Perf3_ModifiedTone_ThreeCharAllTone3_FirstTwoBecomeTone2()
    {
        // 蒙古包: jieba splits as [蒙古|包] (2+1 prosody), all tone 3 →
        // first two become tone 2 (Python parity)
        var finals = new List<string> { "eng3", "u3", "ao3" };
        var result = ToneSandhi.ModifiedTone("蒙古包", "n", finals);
        Assert.Equal('2', result[0][^1]);
        Assert.Equal('2', result[1][^1]);
        Assert.Equal('3', result[2][^1]);
    }

    [Fact]
    public void Perf4_ApplyStress_NoStress_Unchanged()
    {
        var result = EnglishLexicon.ApplyStress("hEloʊ", stress: null);
        Assert.Equal("hEloʊ", result);
    }

    [Fact]
    public void Perf4_ApplyStress_StressUpgrade_MarksPrimary()
    {
        // Add primary stress to a word with no current stress.
        // "hEloʊ" has no stress marker → should gain a primary stress (ˈ) before a vowel.
        var result = EnglishLexicon.ApplyStress("hEloʊ", stress: 2);
        // Result should contain the primary stress character
        Assert.Contains('ˈ', result); // ˈ = primary stress
    }

    [Fact]
    public void Perf4_ApplyStress_RemoveAll()
    {
        // Stress < -1 removes all stress markers
        var result = EnglishLexicon.ApplyStress("ˈhEloʊ", stress: -2);
        Assert.DoesNotContain('ˈ', result);
        Assert.DoesNotContain('ˌ', result);
    }

    [Fact]
    public void Perf4_ApplyStress_NoVowels_UnchangedOnUpgrade()
    {
        // Input with no vowels — restress would have no target, should return unchanged
        var result = EnglishLexicon.ApplyStress("bcd", stress: 2);
        Assert.Equal("bcd", result);
    }

    [Fact]
    public void Perf4_ApplyStress_LongString_StackallocThreshold()
    {
        // Drive the non-stackalloc path (length > 64)
        var input = new string('b', 30) + "a" + new string('c', 40);
        var result = EnglishLexicon.ApplyStress(input, stress: 2);
        // Should contain a primary stress marker placed before the vowel
        Assert.Contains('ˈ', result);
        // And must still contain the vowel
        Assert.Contains('a', result);
    }

    // ==========================================================================
    // Perf 5: ChineseG2P.MapPunctuation — StringBuilder single-pass
    // MapPunctuation is private; tested indirectly through Process.
    // ==========================================================================

    [Fact]
    public void Perf5_MapPunctuation_FullWidthComma_MapsToAscii()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("你好，世界");
        Assert.Contains(",", result.Phonemes);
    }

    [Fact]
    public void Perf5_MapPunctuation_FullWidthPeriod_MapsToAscii()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("你好。");
        Assert.Contains(".", result.Phonemes);
    }

    [Fact]
    public void Perf5_MapPunctuation_Exclamation_MapsToAscii()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("好！");
        Assert.Contains("!", result.Phonemes);
    }

    [Fact]
    public void Perf5_MapPunctuation_Question_MapsToAscii()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("好吗？");
        Assert.Contains("?", result.Phonemes);
    }

    [Fact]
    public void Perf5_MapPunctuation_MultiplePunctuation_AllMapped()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("你好，世界！今天好吗？");
        Assert.Contains(",", result.Phonemes);
        Assert.Contains("!", result.Phonemes);
        Assert.Contains("?", result.Phonemes);
    }

    [Fact]
    public void Perf5_MapPunctuation_Parentheses_Mapped()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("你好（世界）");
        Assert.Contains("(", result.Phonemes);
        Assert.Contains(")", result.Phonemes);
    }

    [Fact]
    public void Perf5_MapPunctuation_NoPunctuation_Passthrough()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("你好世界");
        Assert.NotEmpty(result.Phonemes);
    }

    // ==========================================================================
    // Perf 6: PinyinLookup.GetTone3 — SingleCharDict avoids per-char ToString
    // ==========================================================================

    [Fact]
    public void Perf6_GetTone3_SingleChar_Works()
    {
        var lookup = PinyinLookup.Instance;
        var result = lookup.GetTone3("你");
        Assert.NotEmpty(result);
        // Result should be a pinyin with tone digit
        Assert.Matches(@"^[a-zA-Z]+\d$", result[0]);
    }

    [Fact]
    public void Perf6_GetTone3_MultipleChars_FallsBackToSingleCharLookup()
    {
        var lookup = PinyinLookup.Instance;
        // Use characters unlikely to form a known phrase to exercise per-char fallback
        var result = lookup.GetTone3("你好世界");
        Assert.Equal(4, result.Count);
        foreach (var pinyin in result)
        {
            Assert.False(string.IsNullOrEmpty(pinyin));
        }
    }

    [Fact]
    public void Perf6_GetTone3_UnknownChar_ReturnsEmpty()
    {
        var lookup = PinyinLookup.Instance;
        // ASCII char has no pinyin → empty string
        var result = lookup.GetTone3("A");
        Assert.Single(result);
        Assert.Equal(string.Empty, result[0]);
    }

    [Fact]
    public void Perf6_GetTone3_KnownPhrase_UsesPhraseDict()
    {
        var lookup = PinyinLookup.Instance;
        // 你好 is a common phrase
        var result = lookup.GetTone3("你好");
        Assert.Equal(2, result.Count);
    }
}
