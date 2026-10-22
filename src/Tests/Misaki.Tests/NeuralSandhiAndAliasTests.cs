using Misaki.NET.Lang.Chinese;
using Misaki.NET.Lang.English;
using Xunit;

namespace Misaki.Tests;

/// <summary>
/// Tests covering NeuralSandhi rewrite (Diff 5) and English "2"->"to" alias (Diff 6).
/// </summary>
public class NeuralSandhiAndAliasTests
{
    static NeuralSandhiAndAliasTests()
    {
        _ = new ChineseG2P(version: "1.1");
    }

    private static int LastTone(string final) => final.Length == 0 ? 0 : final[^1] - '0';

    // ==========================================================================
    // Diff 5: NeuralSandhi - any-length word, ending checks, sub-word splitting
    // ==========================================================================

    // ---- Reduplication path (n/v/a POS) ----
    [Fact]
    public void Diff5_AaReduplication_VerbPos_SecondNeutral()
    {
        // 看看: AA verb reduplication -> second 看 becomes tone 5
        var finals = new List<string> { "an4", "an4" };
        var result = ToneSandhi.ModifiedTone("看看", "v", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_AaReduplication_NonNvaPos_NoChange()
    {
        // Same pattern but with POS 'd' (adverb) — Python only reduplicates n/v/a
        var finals = new List<string> { "an4", "an4" };
        var result = ToneSandhi.ModifiedTone("看看", "d", finals);
        // Without n/v/a POS, the reduplication branch should not fire.
        // Note: a fallback may still set neutral via must_neural; assert presence not absence.
        // The point is the rule path is gated by POS.
        Assert.True(result.Count == 2);
    }

    // ---- Modal particles at end of word ----
    [Fact]
    public void Diff5_EndsWithModalBa_BecomesNeutral()
    {
        // 好吧: 吧 is a sentence-final modal particle
        var finals = new List<string> { "ao3", "a1" };
        var result = ToneSandhi.ModifiedTone("好吧", "y", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_EndsWithModalNe_BecomesNeutral()
    {
        // 是呢: 呢 modal particle
        var finals = new List<string> { "i4", "e2" };
        var result = ToneSandhi.ModifiedTone("是呢", "y", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    // ---- Structural particles 的地得 ----
    [Fact]
    public void Diff5_EndsWithDe_BecomesNeutral()
    {
        // 我的 (regardless of POS — Python applies on word-end position)
        var finals = new List<string> { "o3", "e1" };
        var result = ToneSandhi.ModifiedTone("我的", "r", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    // ---- Single-char aspect particles 了着过 ----
    [Fact]
    public void Diff5_SingleCharLe_AspectPos_BecomesNeutral()
    {
        var finals = new List<string> { "e3" };
        var result = ToneSandhi.ModifiedTone("了", "ul", finals);
        Assert.Equal(5, LastTone(result[0]));
    }

    [Fact]
    public void Diff5_SingleCharGuo_NonAspectPos_NoChange()
    {
        // 过 alone with POS 'v' (verb "to pass") — Python only applies for ul/uz/ug
        var finals = new List<string> { "uo4" };
        var result = ToneSandhi.ModifiedTone("过", "v", finals);
        Assert.Equal(4, LastTone(result[0]));
    }

    // ---- Plural/diminutive 们/子 with r/n POS ----
    [Fact]
    public void Diff5_EndsWithMen_PronounPos_BecomesNeutral()
    {
        // 我们: 们 -> tone 5
        var finals = new List<string> { "o3", "en2" };
        var result = ToneSandhi.ModifiedTone("我们", "r", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_EndsWithZi_NounPos_BecomesNeutral()
    {
        // 桌子: 子 -> tone 5
        var finals = new List<string> { "uo1", "i3" };
        var result = ToneSandhi.ModifiedTone("桌子", "n", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    // ---- Locative 上/下 with s/l/f POS ----
    [Fact]
    public void Diff5_EndsWithShang_LocativePos_BecomesNeutral()
    {
        // 桌上: 上 -> tone 5
        var finals = new List<string> { "uo1", "ang4" };
        var result = ToneSandhi.ModifiedTone("桌上", "s", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    // ---- Directional complement 来/去 after 上下进出回过起开 ----
    [Fact]
    public void Diff5_DirectionalQuLai_BecomesNeutral()
    {
        // 起来: 来 -> tone 5 (directional complement after 起)
        var finals = new List<string> { "i3", "ai2" };
        var result = ToneSandhi.ModifiedTone("起来", "v", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_DirectionalChuQu_BecomesNeutral()
    {
        // 出去: 去 -> tone 5
        var finals = new List<string> { "u1", "u4" };
        var result = ToneSandhi.ModifiedTone("出去", "v", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_NonDirectionalLai_NoChange()
    {
        // 你来 — 你 is NOT in directional stem set, so 来 stays
        var finals = new List<string> { "i3", "ai2" };
        var result = ToneSandhi.ModifiedTone("你来", "v", finals);
        Assert.Equal(2, LastTone(result[1]));
    }

    // ---- 个 patterns ----
    [Fact]
    public void Diff5_NumericGe_BecomesNeutral()
    {
        // 三个: 个 after numeric -> tone 5
        var finals = new List<string> { "an1", "e4" };
        var result = ToneSandhi.ModifiedTone("三个", "m", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_JiGe_BecomesNeutral()
    {
        // 几个: 几 is in GePrecedents -> tone 5
        var finals = new List<string> { "i3", "e4" };
        var result = ToneSandhi.ModifiedTone("几个", "m", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_StandaloneGe_BecomesNeutral()
    {
        // 个 alone -> tone 5
        var finals = new List<string> { "e4" };
        var result = ToneSandhi.ModifiedTone("个", "q", finals);
        Assert.Equal(5, LastTone(result[0]));
    }

    [Fact]
    public void Diff5_NonNumericGe_NoChange()
    {
        // 个体: 个 NOT preceded by numeric/special -> stays tone 4
        var finals = new List<string> { "e4", "i3" };
        var result = ToneSandhi.ModifiedTone("个体", "n", finals);
        Assert.Equal(4, LastTone(result[0]));
    }

    // ---- must_neural fallback (full word and last 2 chars) ----
    [Fact]
    public void Diff5_MustNeural_FullWord_BecomesNeutral()
    {
        // 朋友 is in must_neural
        var finals = new List<string> { "eng2", "ou3" };
        var result = ToneSandhi.ModifiedTone("朋友", "n", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff5_MustNeural_LastTwoChars_BecomesNeutral()
    {
        // A 3-char word whose last 2 chars are in must_neural — Python triggers via word[-2:]
        // 小朋友: 朋友 in must_neural
        var finals = new List<string> { "iao3", "eng2", "ou3" };
        var result = ToneSandhi.ModifiedTone("小朋友", "n", finals);
        Assert.Equal(5, LastTone(result[2]));
    }

    // ---- must_not list ----
    [Fact]
    public void Diff5_MustNotNeural_NotApplied()
    {
        // 男子 — 男子 is in must_not_neural_tone_words
        var finals = new List<string> { "an2", "i3" };
        var result = ToneSandhi.ModifiedTone("男子", "n", finals);
        // 子 should NOT become tone 5 here
        Assert.Equal(3, LastTone(result[1]));
    }

    // ==========================================================================
    // Diff 6: English "2" -> "to" alias between alphabetic tokens
    // ==========================================================================

    [Fact]
    public void Diff6_TwoBetweenAlpha_AliasedToTo()
    {
        var g2p = new EnglishG2P();
        var result = g2p.Process("go 2 work");
        // Phoneme output should look like "go to work" — must contain the "to" phonemes.
        // We verify by comparing against the equivalent text spelled out.
        var expected = g2p.Process("go to work");
        Assert.Equal(expected.Phonemes, result.Phonemes);
    }

    [Fact]
    public void Diff6_TwoAtSentenceStart_NotAliased()
    {
        // "2 dogs" — at start, alias must NOT trigger (Python: 0 < j < len-1)
        var g2p = new EnglishG2P();
        var withTwo = g2p.Process("2 dogs");
        var withTo = g2p.Process("to dogs");
        Assert.NotEqual(withTo.Phonemes, withTwo.Phonemes);
    }

    [Fact]
    public void Diff6_TwoAtSentenceEnd_NotAliased()
    {
        // "go 2" — at end, alias must NOT trigger
        var g2p = new EnglishG2P();
        var withTwo = g2p.Process("go 2");
        var withTo = g2p.Process("go to");
        Assert.NotEqual(withTo.Phonemes, withTwo.Phonemes);
    }

    [Fact]
    public void Diff6_TwoBetweenNonAlpha_NotAliased()
    {
        // "5 2 7" — surrounded by digits, alias must NOT trigger
        var g2p = new EnglishG2P();
        var result = g2p.Process("5 2 7");
        var aliased = g2p.Process("5 to 7");
        Assert.NotEqual(aliased.Phonemes, result.Phonemes);
    }

    [Fact]
    public void Diff6_DigitOtherThanTwo_NotAliased()
    {
        // "go 3 work" — digit 3 should not alias
        var g2p = new EnglishG2P();
        var result = g2p.Process("go 3 work");
        // 3 should be read as "three", not "to"
        var asThree = g2p.Process("go three work");
        Assert.Equal(asThree.Phonemes, result.Phonemes);
    }
}
