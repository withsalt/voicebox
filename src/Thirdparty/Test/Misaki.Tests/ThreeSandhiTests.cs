using Misaki.NET.Lang.Chinese;
using Xunit;

namespace Misaki.Tests;

/// <summary>
/// Extensive tests for the rewritten ThreeSandhi (third-tone sandhi) method.
/// Tone 3 + tone 3 → tone 2 + tone 3, with prosody-aware handling for 3-char words
/// and 2+2 split for 4-char words. Mirrors Python tone_sandhi._three_sandhi.
/// </summary>
public class ThreeSandhiTests
{
    static ThreeSandhiTests()
    {
        _ = new ChineseG2P(version: "1.1");
    }

    private static int LastTone(string final) => final.Length == 0 ? 0 : final[^1] - '0';

    private static List<string> InvokeThreeSandhi(string word, List<string> finals)
    {
        // ModifiedTone applies all 4 sandhi rules. Use a POS that won't trigger
        // BuSandhi/YiSandhi/NeuralSandhi (must_neural fallbacks).
        // Skip BuSandhi/YiSandhi by using words without 不/一.
        return ToneSandhi.ModifiedTone(word, "x", finals);
    }

    // ==========================================================================
    // 2-char words
    // ==========================================================================

    [Fact]
    public void TwoChar_AllToneThree_FirstBecomesTwo()
    {
        // 你好: ni3 + hao3 → ni2 + hao3
        var finals = new List<string> { "i3", "ao3" };
        var result = InvokeThreeSandhi("您好", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(3, LastTone(result[1]));
    }

    [Fact]
    public void TwoChar_FirstNotToneThree_NoChange()
    {
        // 你来: ni3 + lai2 → no change
        var finals = new List<string> { "i3", "ai2" };
        var result = InvokeThreeSandhi("你来", finals);
        Assert.Equal(3, LastTone(result[0]));
        Assert.Equal(2, LastTone(result[1]));
    }

    [Fact]
    public void TwoChar_SecondNotToneThree_NoChange()
    {
        // 来了: lai2 + le3 (artificial example)
        var finals = new List<string> { "ai2", "ao3" };
        var result = InvokeThreeSandhi("来好", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(3, LastTone(result[1]));
    }

    [Fact]
    public void TwoChar_NeitherToneThree_NoChange()
    {
        var finals = new List<string> { "ai1", "an2" };
        var result = InvokeThreeSandhi("来年", finals);
        Assert.Equal(1, LastTone(result[0]));
        Assert.Equal(2, LastTone(result[1]));
    }

    // ==========================================================================
    // 3-char words: 2+1 prosody (jieba splits as [AB|C])
    // ==========================================================================

    [Fact]
    public void ThreeChar_2plus1_AllToneThree_BothFirstBecomeTwo()
    {
        // 蒙古包: jieba splits as [蒙古|包] (sub1Len=2). All tone 3 →
        // finals[0]=2, finals[1]=2, finals[2]=3
        var finals = new List<string> { "eng3", "u3", "ao3" };
        var result = InvokeThreeSandhi("蒙古包", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(2, LastTone(result[1]));
        Assert.Equal(3, LastTone(result[2]));
    }

    [Fact]
    public void ThreeChar_2plus1_AllToneThree_AnotherCase()
    {
        // 所有人: jieba splits as [所有|人] (sub1Len=2)
        // Note: 所有人 finals are likely so3+you3+ren2, not all-tone-3.
        // For all-tone-3 test we use the artificial case: 木头人 with all-3 finals
        // Actually let's craft a clean all-tone-3 example: 我管你 (artificial)
        // Use 蒙古包-style synthetic finals
        var finals = new List<string> { "uo3", "ou3", "en3" };
        // 所有人 is split [所有|人] by jieba (sub1Len=2)
        var result = InvokeThreeSandhi("所有人", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(2, LastTone(result[1]));
        Assert.Equal(3, LastTone(result[2]));
    }

    [Fact]
    public void ThreeChar_2plus1_Mixed_Sub1AllToneThree_FirstBecomesTwo()
    {
        // 所有人 (real finals): sub1=所有 (suo3+you3, all-tone-3), sub2=人 (ren2)
        // Per Python rule: sub1 is len 2 and all-tone-3 → finals[0] becomes 2
        var finals = new List<string> { "uo3", "ou3", "en2" };
        var result = InvokeThreeSandhi("所有人", finals);
        Assert.Equal(2, LastTone(result[0])); // 所 → 2
        Assert.Equal(3, LastTone(result[1])); // 有 stays 3 (sub2 is len 1, no internal change)
        Assert.Equal(2, LastTone(result[2])); // 人 unchanged
    }

    [Fact]
    public void ThreeChar_2plus1_Mixed_Sub1NotAllToneThree_NoChange()
    {
        // 蒙古包 with finals meng2+gu3+bao1 (artificial): sub1=蒙古 not all-3
        // → no changes
        var finals = new List<string> { "eng2", "u3", "ao1" };
        var result = InvokeThreeSandhi("蒙古包", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(3, LastTone(result[1]));
        Assert.Equal(1, LastTone(result[2]));
    }

    // ==========================================================================
    // 3-char words: 1+2 prosody (jieba splits as [A|BC])
    // ==========================================================================

    [Fact]
    public void ThreeChar_1plus2_AllToneThree_SecondBecomesTwo()
    {
        // 纸老虎: jieba splits as [纸|老虎] (sub1Len=1). All tone 3 →
        // 1+2 prosody: finals[1]=2 (the second char becomes tone 2 within "老虎")
        var finals = new List<string> { "i3", "ao3", "u3" };
        var result = InvokeThreeSandhi("纸老虎", finals);
        Assert.Equal(3, LastTone(result[0])); // 纸 stays
        Assert.Equal(2, LastTone(result[1])); // 老 → 2
        Assert.Equal(3, LastTone(result[2])); // 虎 stays
    }

    [Fact]
    public void ThreeChar_1plus2_BoundaryToneThree_TriggersTransition()
    {
        // Test the Python rule: i==1, sub not all-tone-3, sub[0] is tone 3,
        // and first sub's last is tone 3 → first sub's last becomes 2.
        // Use a word where jieba splits 1+2 and the boundary has tone 3 on both sides.
        // 我很好: jieba splits [我|很好] (sub1Len=1). Finals: wo3+hen3+hao3 (all tone 3)
        // → all-tone-3 branch fires instead. Need a mixed case.
        // Use artificial: word "好你们" with finals hao3+ni3+men2
        // jieba may split differently. Let's use a direct approach:
        // 小朋友 splits [小|朋友]. Finals: xiao3+peng2+you3 (not all-tone-3)
        // sub1=["xiao3"], sub2=["peng2","you3"]. sub2 not all-tone-3.
        // i==1: sub2[0] tone=2 (not 3) → condition fails. No change.
        // We need sub2[0] to be tone 3. Let's use 好你来: hao3+ni3+lai2
        // jieba splits [好|你来] (sub1Len=1). sub2=["ni3","lai2"] not all-tone-3.
        // i==1: sub2[0] tone=3, finalsList[0][^1] (=hao3) tone=3 → trigger!
        // finalsList[0][^1] = hao3 → hao2
        var finals = new List<string> { "ao3", "i3", "ai2" };
        var result = ToneSandhi.ModifiedTone("好你来", "x", finals);
        Assert.Equal(2, LastTone(result[0])); // 好 → 2 (transitional sandhi)
        Assert.Equal(3, LastTone(result[1])); // 你 stays
        Assert.Equal(2, LastTone(result[2])); // 来 stays
    }

    [Fact]
    public void ThreeChar_1plus2_Sub2AllToneThree_LenTwo_FirstOfSub2BecomesTwo()
    {
        // Synthetic 1+2 with sub2 all-tone-3: e.g. "好你好" splits [好|你好]
        // Wait, jieba behavior depends on dict; let's use 我很好 which jieba splits [我|很好]
        // Finals: wo3 + hen3 + hao3 (all tone 3)
        // sub1Len=1 → 1+2 prosody, all-tone-3 → finals[1]=2
        var finals = new List<string> { "o3", "en3", "ao3" };
        var result = InvokeThreeSandhi("我很好", finals);
        Assert.Equal(3, LastTone(result[0])); // 我 stays
        Assert.Equal(2, LastTone(result[1])); // 很 → 2
        Assert.Equal(3, LastTone(result[2])); // 好 stays
    }

    [Fact]
    public void ThreeChar_NoToneThree_NoChange()
    {
        // 蒙古人 with all non-3 finals
        var finals = new List<string> { "eng2", "u4", "en2" };
        var result = InvokeThreeSandhi("蒙古人", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(4, LastTone(result[1]));
        Assert.Equal(2, LastTone(result[2]));
    }

    // ==========================================================================
    // 4-char words: 2+2 split, each pair handled independently
    // ==========================================================================

    [Fact]
    public void FourChar_FirstPairAllToneThree_FirstOfFirstPairBecomesTwo()
    {
        // [我好|平安] (artificial): wo3+hao3+ping2+an1
        // First pair all-tone-3 → finals[0]=2; second pair not all-tone-3 → no change
        var finals = new List<string> { "o3", "ao3", "ing2", "an1" };
        var result = InvokeThreeSandhi("我好平安", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(3, LastTone(result[1]));
        Assert.Equal(2, LastTone(result[2]));
        Assert.Equal(1, LastTone(result[3]));
    }

    [Fact]
    public void FourChar_SecondPairAllToneThree_FirstOfSecondPairBecomesTwo()
    {
        // [平安|我好] (artificial): ping2+an1+wo3+hao3
        var finals = new List<string> { "ing2", "an1", "o3", "ao3" };
        var result = InvokeThreeSandhi("平安我好", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(1, LastTone(result[1]));
        Assert.Equal(2, LastTone(result[2])); // 我 → 2
        Assert.Equal(3, LastTone(result[3]));
    }

    [Fact]
    public void FourChar_BothPairsAllToneThree_BothFirstsBecomeTwo()
    {
        // [我好|你好] (artificial): wo3+hao3+ni3+hao3
        var finals = new List<string> { "o3", "ao3", "i3", "ao3" };
        var result = InvokeThreeSandhi("我好你好", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(3, LastTone(result[1]));
        Assert.Equal(2, LastTone(result[2]));
        Assert.Equal(3, LastTone(result[3]));
    }

    [Fact]
    public void FourChar_PairwiseAcrossSplitNotApplied()
    {
        // [来好|你来] artificial: lai2+hao3+ni3+lai2
        // Position 1+2 (hao3+ni3) are both tone 3 ACROSS the 2+2 split
        // Python's 2+2 split rule treats each pair independently — the
        // boundary tone-3 sequence should NOT trigger a sandhi change.
        var finals = new List<string> { "ai2", "ao3", "i3", "ai2" };
        var result = InvokeThreeSandhi("来好你来", finals);
        Assert.Equal(2, LastTone(result[0]));
        Assert.Equal(3, LastTone(result[1])); // 好 stays — first pair not all-tone-3
        Assert.Equal(3, LastTone(result[2])); // 你 stays — second pair not all-tone-3
        Assert.Equal(2, LastTone(result[3]));
    }

    [Fact]
    public void FourChar_NoToneThree_NoChange()
    {
        var finals = new List<string> { "an1", "an2", "an1", "an2" };
        var result = InvokeThreeSandhi("安年安年", finals);
        Assert.Equal(1, LastTone(result[0]));
        Assert.Equal(2, LastTone(result[1]));
        Assert.Equal(1, LastTone(result[2]));
        Assert.Equal(2, LastTone(result[3]));
    }

    // ==========================================================================
    // 5+ char words and 1-char words: Python performs no changes
    // ==========================================================================

    [Fact]
    public void FiveChar_NoChange()
    {
        // Python only handles len 2/3/4. Length 5 → no tone-3 sandhi.
        var finals = new List<string> { "o3", "ao3", "i3", "ao3", "i3" };
        var result = InvokeThreeSandhi("我好你好你", finals);
        // All tone-3s should stay tone 3
        Assert.Equal(3, LastTone(result[0]));
        Assert.Equal(3, LastTone(result[1]));
        Assert.Equal(3, LastTone(result[2]));
        Assert.Equal(3, LastTone(result[3]));
        Assert.Equal(3, LastTone(result[4]));
    }

    [Fact]
    public void OneChar_NoChange()
    {
        // Single-char input: no tone-3 sandhi possible
        var finals = new List<string> { "ao3" };
        var result = InvokeThreeSandhi("好", finals);
        Assert.Equal(3, LastTone(result[0]));
    }

    [Fact]
    public void EmptyFinals_NoCrash()
    {
        var result = InvokeThreeSandhi("", new List<string>());
        Assert.Empty(result);
    }
}
