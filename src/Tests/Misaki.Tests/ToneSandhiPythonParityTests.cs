using Misaki.NET.Lang.Chinese;
using Xunit;

namespace Misaki.Tests;

/// <summary>
/// Tests covering the first-layer Python parity fixes for tone sandhi
/// (YiSandhi, MergeYi, MergeReduplication, MergeContinuousThreeTones).
/// These verify the rule logic matches the Python reference implementation.
/// </summary>
public class ToneSandhiPythonParityTests
{
    // Make sure the bridges are initialized before tests that depend on
    // PinyinLookupBridge or JiebaBridge run.
    static ToneSandhiPythonParityTests()
    {
        _ = new ChineseG2P(version: "1.1");
    }

    private static int LastTone(string final) => final.Length == 0 ? 0 : final[^1] - '0';

    // ==========================================================================
    // Diff 1a: YiSandhi - 第一X (e.g. 第一次, 第一个) keeps yi1
    // Old code: word == "第一" exact match
    // New code: word.StartsWith("第一")
    // ==========================================================================

    [Fact]
    public void Diff1a_DiYi_Exact_KeepsYi1()
    {
        var finals = new List<string> { "i4", "i1" };
        var result = ToneSandhi.ModifiedTone("第一", "m", finals);
        Assert.Equal(1, LastTone(result[1])); // 一 stays tone 1
    }

    [Fact]
    public void Diff1a_DiYiCi_KeepsYi1()
    {
        // 第一次: 第(i4) 一(i1) 次(i4) — without the fix, 一 would change to yi2 before tone 4
        var finals = new List<string> { "i4", "i1", "i4" };
        var result = ToneSandhi.ModifiedTone("第一次", "m", finals);
        Assert.Equal(1, LastTone(result[1])); // 一 stays tone 1
    }

    [Fact]
    public void Diff1a_DiYiGe_KeepsYi1()
    {
        // 第一个
        var finals = new List<string> { "i4", "i1", "e4" };
        var result = ToneSandhi.ModifiedTone("第一个", "m", finals);
        Assert.Equal(1, LastTone(result[1]));
    }

    // ==========================================================================
    // Diff 1b: YiSandhi - tone 4 OR tone 5 trigger
    // Old code: only tone 4 triggered yi -> yi2
    // New code: tone 4 or tone 5 (neutral) triggers yi -> yi2
    // ==========================================================================

    [Fact]
    public void Diff1b_YiBeforeTone4_BecomesTone2()
    {
        // 一样: yi1 + ang4 -> yi2 + ang4
        var finals = new List<string> { "i1", "ang4" };
        var result = ToneSandhi.ModifiedTone("一样", "d", finals);
        Assert.Equal(2, LastTone(result[0]));
    }

    [Fact]
    public void Diff1b_YiBeforeTone5_BecomesTone2()
    {
        // 一个 (the suffix-个 may be neutral tone 5): yi -> yi2
        var finals = new List<string> { "i1", "e5" };
        var result = ToneSandhi.ModifiedTone("一个", "m", finals);
        Assert.Equal(2, LastTone(result[0]));
    }

    [Fact]
    public void Diff1b_YiBeforeTone1_BecomesTone4()
    {
        // 一天: yi1 + ian1 -> yi4 + ian1
        var finals = new List<string> { "i1", "ian1" };
        var result = ToneSandhi.ModifiedTone("一天", "m", finals);
        Assert.Equal(4, LastTone(result[0]));
    }

    [Fact]
    public void Diff1b_YiBeforeTone2_BecomesTone4()
    {
        // 一年
        var finals = new List<string> { "i1", "ian2" };
        var result = ToneSandhi.ModifiedTone("一年", "m", finals);
        Assert.Equal(4, LastTone(result[0]));
    }

    [Fact]
    public void Diff1b_YiBeforeTone3_BecomesTone4()
    {
        // 一起: 起 is tone 3 and not a numeric, so Yi sandhi triggers normally → yi4
        var finals = new List<string> { "i1", "i3" };
        var result = ToneSandhi.ModifiedTone("一起", "d", finals);
        Assert.Equal(4, LastTone(result[0]));
    }

    // ==========================================================================
    // Diff 1c: YiSandhi - pure digit sequence skip
    // 一 in pure digit/numeric sequences (一零零, 二一三) should NOT change tone
    // ==========================================================================

    [Fact]
    public void Diff1c_PureDigitSequence_YiUnchanged()
    {
        // 一零零: all chars are 一 + numerics -> no tone change
        var finals = new List<string> { "i1", "ing2", "ing2" };
        var result = ToneSandhi.ModifiedTone("一零零", "m", finals);
        // Without the fix, 一 would change to yi2 before tone 2
        Assert.Equal(1, LastTone(result[0]));
    }

    [Fact]
    public void Diff1c_NumericPrefixYi_Unchanged()
    {
        // 二一三: 二 + 一 + 三, all numeric -> no tone change
        var finals = new List<string> { "er4", "i1", "an1" };
        var result = ToneSandhi.ModifiedTone("二一三", "m", finals);
        Assert.Equal(1, LastTone(result[1]));
    }

    [Fact]
    public void Diff1c_MixedNonNumeric_AppliesNormalSandhi()
    {
        // 一样 contains 样 (non-numeric) -> normal sandhi applies
        var finals = new List<string> { "i1", "ang4" };
        var result = ToneSandhi.ModifiedTone("一样", "d", finals);
        Assert.Equal(2, LastTone(result[0]));
    }

    // ==========================================================================
    // Diff 1d: YiSandhi - V一V reduplication (length-3, word[0]==word[-1]) -> yi5
    // Mirrors Python: if len(word) == 3 and word[1] == YI and word[0] == word[-1]: yi5
    // ==========================================================================

    [Fact]
    public void Diff1d_VYiV_Reduplication_BecomesNeutral()
    {
        // 看一看: 一 should become tone 5
        var finals = new List<string> { "an4", "i1", "an4" };
        var result = ToneSandhi.ModifiedTone("看一看", "v", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void Diff1d_NonVYiV_NotNeutral()
    {
        // 第一次 (length 3 but word[0] != word[-1]) -> 一 stays as 第一 logic
        var finals = new List<string> { "i4", "i1", "i4" };
        var result = ToneSandhi.ModifiedTone("第一次", "m", finals);
        Assert.Equal(1, LastTone(result[1]));
    }

    // ==========================================================================
    // Diff 2: MergeYi requires previous POS == "v" for V一V pattern
    // Without the POS check, "X一X" with non-verbs would also be merged.
    // ==========================================================================

    [Fact]
    public void Diff2_MergeYi_VerbV1V_Merges()
    {
        // 看一看 with 看 as verb -> merged into single segment
        var input = new List<(string, string)> { ("看", "v"), ("一", "m"), ("看", "v") };
        var result = ToneSandhi.PreMergeForModify(input);
        // Should be one merged segment
        Assert.Single(result);
        Assert.Equal("看一看", result[0].word);
    }

    [Fact]
    public void Diff2_MergeYi_NonVerb1Pattern_NotMerged()
    {
        // 人一人 with 人 as noun (n) — must NOT be merged into V一V
        // The "一人" subsequence may still merge into a single segment via the second branch,
        // but the V一V triple-merge must NOT trigger for non-verbs.
        var input = new List<(string, string)> { ("人", "n"), ("一", "m"), ("人", "n") };
        var result = ToneSandhi.PreMergeForModify(input);
        // The V一V merge requires pos=="v"; with "n" it must not produce a single 3-char segment
        Assert.NotEqual("人一人", result[0].word);
    }

    // ==========================================================================
    // Diff 3: MergeReduplication supports any-length identical adjacent words
    // Old: only single-char A+A merged. New: any length where word == previous.
    // ==========================================================================

    [Fact]
    public void Diff3_MergeReduplication_SingleChar_StillMerges()
    {
        var input = new List<(string, string)> { ("看", "v"), ("看", "v") };
        var result = ToneSandhi.PreMergeForModify(input);
        Assert.Single(result);
        Assert.Equal("看看", result[0].word);
    }

    [Fact]
    public void Diff3_MergeReduplication_TwoChar_NowMerges()
    {
        // 研究研究 (two-char verb reduplication) — Python merges this; old .NET did not
        var input = new List<(string, string)> { ("研究", "v"), ("研究", "v") };
        var result = ToneSandhi.PreMergeForModify(input);
        Assert.Single(result);
        Assert.Equal("研究研究", result[0].word);
    }

    [Fact]
    public void Diff3_MergeReduplication_DifferentWords_NotMerged()
    {
        var input = new List<(string, string)> { ("看", "v"), ("听", "v") };
        var result = ToneSandhi.PreMergeForModify(input);
        Assert.Equal(2, result.Count);
    }

    // ==========================================================================
    // Diff 4: MergeContinuousThreeTones merge_last cascade prevention
    // After two segments are merged (segment B merged into A), segment C cannot
    // also be merged into AB even if all are tone 3.
    // ==========================================================================

    [Fact]
    public void Diff4_MergeContinuousThreeTones_NoCascade()
    {
        // Three single-char tone-3 segments. Pass 1 merges A+B with cascade guard,
        // then pass 2 (MergeContinuousThreeTones2) merges (AB)+C since len 2+1=3 ≤ 3.
        // This is the correct Python parity behavior — a single merged 3-char segment.
        var input = new List<(string, string)> { ("我", "r"), ("很", "d"), ("好", "a") };
        var result = ToneSandhi.PreMergeForModify(input);
        // Final result: one segment "我很好"
        Assert.Single(result);
        Assert.Equal("我很好", result[0].word);
    }

    [Fact]
    public void Diff4_MergeContinuousThreeTones_FourSegments_NoOverMerge()
    {
        // Four single-char tone-3 segments. Cascade prevention ensures we never
        // produce a 4-char merge (Python guard: prev.Length + cur.Length <= 3).
        var input = new List<(string, string)> { ("我", "r"), ("很", "d"), ("好", "a"), ("小", "a") };
        var result = ToneSandhi.PreMergeForModify(input);
        // Must NOT produce a single 4-char merge
        foreach (var (word, _) in result)
        {
            Assert.True(word.Length <= 3, $"Segment '{word}' exceeds 3 chars (cascade not prevented)");
        }
    }

    [Fact]
    public void Diff4_MergeContinuousThreeTones_TwoSegmentMergeStillWorks()
    {
        var input = new List<(string, string)> { ("我", "r"), ("好", "a") };
        var result = ToneSandhi.PreMergeForModify(input);
        Assert.Single(result);
        Assert.Equal("我好", result[0].word);
    }

    [Fact]
    public void Diff4_MergeContinuousThreeTones_NotAllTone3_NoMerge()
    {
        // 我 (wo3) + 来 (lai2) — second is tone 2, no merge
        var input = new List<(string, string)> { ("我", "r"), ("来", "v") };
        var result = ToneSandhi.PreMergeForModify(input);
        Assert.Equal(2, result.Count);
    }
}
