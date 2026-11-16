using Misaki.NET.Lang.Chinese;
using Xunit;

namespace Misaki.Tests;

/// <summary>
/// Extensive tests for the rewritten NeuralSandhi (neutral tone sandhi) method.
/// Covers all 9 rule branches, must_not/must_neural lists, and sub-word splitting.
/// Mirrors Python tone_sandhi._neural_sandhi.
/// </summary>
public class NeuralSandhiExtensiveTests
{
    static NeuralSandhiExtensiveTests()
    {
        _ = new ChineseG2P(version: "1.1");
    }

    private static int LastTone(string final) => final.Length == 0 ? 0 : final[^1] - '0';

    // ==========================================================================
    // Rule 0: must_not_neural_tone_words — early return, no changes
    // ==========================================================================

    [Theory]
    [InlineData("男子")]
    [InlineData("女子")]
    [InlineData("分子")]
    [InlineData("电子")]
    [InlineData("人人")]
    [InlineData("哈哈")]
    public void MustNot_NoNeutralApplied(string word)
    {
        var finals = new List<string> { "an2", "i3" };
        var result = ToneSandhi.ModifiedTone(word, "n", finals);
        Assert.NotEqual(5, LastTone(result[1]));
    }

    // ==========================================================================
    // Rule 1: Reduplication (n/v/a POS) — word[j]==word[j-1] → tone 5
    // ==========================================================================

    [Theory]
    [InlineData("看看", "v")]
    [InlineData("试试", "v")]
    [InlineData("奶奶", "n")]
    [InlineData("旺旺", "a")]
    public void Reduplication_NVA_SecondBecomesNeutral(string word, string pos)
    {
        var finals = new List<string> { "an4", "an4" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Theory]
    [InlineData("看看", "d")]
    [InlineData("看看", "p")]
    public void Reduplication_NonNVA_NoReduplicationChange(string word, string pos)
    {
        var finals = new List<string> { "an4", "an4" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        // Without n/v/a POS, the reduplication rule should not fire
        Assert.NotEqual(5, LastTone(result[1]));
    }

    [Fact]
    public void Reduplication_ThreeChar_MiddleRepeats()
    {
        // 妈妈妈 (artificial): second and third repeat → both become 5
        var finals = new List<string> { "a1", "a1", "a1" };
        var result = ToneSandhi.ModifiedTone("妈妈妈", "n", finals);
        Assert.Equal(5, LastTone(result[1]));
        Assert.Equal(5, LastTone(result[2]));
    }

    // ==========================================================================
    // Rule 2: Modal particles at end of word
    // ==========================================================================

    [Theory]
    [InlineData("好吧", "吧")]
    [InlineData("是呢", "呢")]
    [InlineData("好啊", "啊")]
    [InlineData("好嘛", "嘛")]
    [InlineData("好哦", "哦")]
    [InlineData("好哒", "哒")]
    [InlineData("好耶", "耶")]
    public void ModalParticle_LastBecomesNeutral(string word, string particle)
    {
        ArgumentNullException.ThrowIfNull(particle);

        var finals = new List<string> { "ao3", "a1" };
        var result = ToneSandhi.ModifiedTone(word, "y", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    // ==========================================================================
    // Rule 3: Structural particles 的地得
    // ==========================================================================

    [Theory]
    [InlineData("我的")]
    [InlineData("快地")]
    [InlineData("跑得")]
    public void StructuralParticle_LastBecomesNeutral(string word)
    {
        var finals = new List<string> { "o3", "e1" };
        var result = ToneSandhi.ModifiedTone(word, "r", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    // ==========================================================================
    // Rule 4: Aspect particles 了着过 (single-char, specific POS)
    // ==========================================================================

    [Theory]
    [InlineData("了", "ul")]
    [InlineData("着", "uz")]
    [InlineData("过", "ug")]
    public void AspectParticle_SingleChar_CorrectPOS_BecomesNeutral(string word, string pos)
    {
        var finals = new List<string> { "e3" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        Assert.Equal(5, LastTone(result[0]));
    }

    [Theory]
    [InlineData("了", "v")]
    [InlineData("过", "v")]
    [InlineData("着", "n")]
    public void AspectParticle_WrongPOS_NoChange(string word, string pos)
    {
        var finals = new List<string> { "e3" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        Assert.Equal(3, LastTone(result[0]));
    }

    // ==========================================================================
    // Rule 5: Plural/diminutive 们/子 with r/n POS
    // ==========================================================================

    [Theory]
    [InlineData("我们", "r")]
    [InlineData("他们", "r")]
    [InlineData("桌子", "n")]
    [InlineData("孩子", "n")]
    public void PluralDiminutive_CorrectPOS_LastBecomesNeutral(string word, string pos)
    {
        var finals = new List<string> { "o3", "en2" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Theory]
    [InlineData("我们", "v")]
    [InlineData("桌子", "v")]
    public void PluralDiminutive_WrongPOS_NoChange(string word, string pos)
    {
        var finals = new List<string> { "o3", "en2" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        Assert.NotEqual(5, LastTone(result[1]));
    }

    // ==========================================================================
    // Rule 6: Locative 上/下 with s/l/f POS
    // ==========================================================================

    [Theory]
    [InlineData("桌上", "s")]
    [InlineData("地下", "f")]
    [InlineData("楼上", "l")]
    public void Locative_CorrectPOS_LastBecomesNeutral(string word, string pos)
    {
        var finals = new List<string> { "uo1", "ang4" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Theory]
    [InlineData("桌上", "n")]
    [InlineData("地下", "v")]
    public void Locative_WrongPOS_NoChange(string word, string pos)
    {
        var finals = new List<string> { "uo1", "ang4" };
        var result = ToneSandhi.ModifiedTone(word, pos, finals);
        Assert.NotEqual(5, LastTone(result[1]));
    }

    // ==========================================================================
    // Rule 7: Directional complement 来/去 after 上下进出回过起开
    // ==========================================================================

    [Theory]
    [InlineData("起来")]
    [InlineData("出去")]
    [InlineData("上来")]
    [InlineData("下去")]
    [InlineData("进来")]
    [InlineData("回来")]
    [InlineData("过来")]
    [InlineData("开来")]
    public void DirectionalComplement_LastBecomesNeutral(string word)
    {
        var finals = new List<string> { "i3", "ai2" };
        var result = ToneSandhi.ModifiedTone(word, "v", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Theory]
    [InlineData("你来")]
    [InlineData("我去")]
    [InlineData("他来")]
    public void NonDirectional_NoChange(string word)
    {
        var finals = new List<string> { "i3", "ai2" };
        var result = ToneSandhi.ModifiedTone(word, "v", finals);
        Assert.NotEqual(5, LastTone(result[1]));
    }

    // ==========================================================================
    // Rule 8: 个 as quantifier
    // ==========================================================================

    [Theory]
    [InlineData("三个")]
    [InlineData("五个")]
    [InlineData("几个")]
    [InlineData("有个")]
    [InlineData("两个")]
    [InlineData("半个")]
    [InlineData("多个")]
    [InlineData("各个")]
    [InlineData("整个")]
    [InlineData("每个")]
    [InlineData("做个")]
    [InlineData("是个")]
    public void GeQuantifier_BecomesNeutral(string word)
    {
        var finals = new List<string> { "an1", "e4" };
        var result = ToneSandhi.ModifiedTone(word, "m", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void GeStandalone_BecomesNeutral()
    {
        var finals = new List<string> { "e4" };
        var result = ToneSandhi.ModifiedTone("个", "q", finals);
        Assert.Equal(5, LastTone(result[0]));
    }

    [Fact]
    public void GeNotPrecededByNumeric_NoChange()
    {
        // 个体: 个 at position 0, no preceding char → no quantifier rule
        var finals = new List<string> { "e4", "i3" };
        var result = ToneSandhi.ModifiedTone("个体", "n", finals);
        Assert.Equal(4, LastTone(result[0]));
    }

    // ==========================================================================
    // Rule 9: must_neural_tone_words fallback (full word + last 2 chars)
    // ==========================================================================

    [Theory]
    [InlineData("朋友")]
    [InlineData("东西")]
    [InlineData("什么")]
    [InlineData("意思")]
    [InlineData("时候")]
    [InlineData("明白")]
    [InlineData("漂亮")]
    [InlineData("清楚")]
    public void MustNeural_FullWord_LastBecomesNeutral(string word)
    {
        var finals = new List<string> { "eng2", "ou3" };
        var result = ToneSandhi.ModifiedTone(word, "n", finals);
        Assert.Equal(5, LastTone(result[1]));
    }

    [Fact]
    public void MustNeural_LastTwoChars_ThreeCharWord()
    {
        // 小朋友: last 2 chars "朋友" in must_neural → last final becomes 5
        var finals = new List<string> { "iao3", "eng2", "ou3" };
        var result = ToneSandhi.ModifiedTone("小朋友", "n", finals);
        Assert.Equal(5, LastTone(result[2]));
    }

    [Fact]
    public void MustNeural_SubWordSplit_FirstSubInMustNeural()
    {
        // 朋友们: jieba splits [朋友|们]. Sub1="朋友" in must_neural →
        // result[sub1Len-1] = result[1] becomes 5
        var finals = new List<string> { "eng2", "ou3", "en2" };
        var result = ToneSandhi.ModifiedTone("朋友们", "n", finals);
        Assert.Equal(5, LastTone(result[1])); // 友 → 5 (sub1 must_neural)
    }

    // ==========================================================================
    // Edge cases
    // ==========================================================================

    [Fact]
    public void EmptyWord_NoCrash()
    {
        var result = ToneSandhi.ModifiedTone("", "n", new List<string>());
        Assert.Empty(result);
    }

    [Fact]
    public void SingleChar_NonSpecial_NoChange()
    {
        var finals = new List<string> { "ao3" };
        var result = ToneSandhi.ModifiedTone("好", "a", finals);
        Assert.Equal(3, LastTone(result[0]));
    }

    [Fact]
    public void NullPos_NoCrash()
    {
        var finals = new List<string> { "ao3", "ao3" };
        var result = ToneSandhi.ModifiedTone("好好", "", finals);
        Assert.Equal(2, result.Count);
    }
}
