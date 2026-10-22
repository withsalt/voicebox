using System.Reflection;
using Misaki.NET.Lang.Chinese;
using Xunit;

namespace Misaki.Tests;

/// <summary>
/// Tests covering #6 (PinyinLookup.GetInitialsAndFinals — single-pass lookup),
/// #7 (ToneSandhi.PreMergeForModify with a shared GetWordFinals cache), and
/// #8 (TextNormalizer traditional→simplified map built from deduped literals).
///
/// The goal for each: behaviour must be identical to the pre-refactor code paths.
/// </summary>
public class ChineseOptimizationTests
{
    // =========================================================================
    // #6 — GetInitialsAndFinals returns exactly the same pair of lists as
    //       calling GetInitials and GetFinals independently.
    // =========================================================================

    [Fact]
    public void GetInitialsAndFinals_EqualsSeparateCalls_CommonWord()
    {
        var lookup = PinyinLookup.Instance;
        var expectedI = lookup.GetInitials("中国");
        var expectedF = lookup.GetFinals("中国");

        var (i, f) = lookup.GetInitialsAndFinals("中国");
        Assert.Equal(expectedI, i);
        Assert.Equal(expectedF, f);
    }

    [Fact]
    public void GetInitialsAndFinals_EqualsSeparateCalls_LongerPhrase()
    {
        var lookup = PinyinLookup.Instance;
        const string word = "我们都是中国人";
        var expectedI = lookup.GetInitials(word);
        var expectedF = lookup.GetFinals(word);

        var (i, f) = lookup.GetInitialsAndFinals(word);
        Assert.Equal(expectedI, i);
        Assert.Equal(expectedF, f);
        // Counts should equal the input length.
        Assert.Equal(word.Length, i.Count);
        Assert.Equal(word.Length, f.Count);
    }

    [Fact]
    public void GetInitialsAndFinals_EnSpecialCase_Preserved()
    {
        // 嗯 has a special final "n2" in GetFinals; GetInitialsAndFinals must preserve that.
        var lookup = PinyinLookup.Instance;
        var (_, f) = lookup.GetInitialsAndFinals("嗯");
        Assert.Equal("n2", f[0]);
    }

    [Fact]
    public void GetInitialsAndFinals_UnknownChar_ProducesEmptyEntry()
    {
        var lookup = PinyinLookup.Instance;
        var (i, f) = lookup.GetInitialsAndFinals("A"); // ASCII → no pinyin
        Assert.Single(i);
        Assert.Single(f);
        Assert.Equal(string.Empty, i[0]);
        Assert.Equal(string.Empty, f[0]);
    }

    [Fact]
    public void GetInitialsAndFinals_EmptyInput_ReturnsEmpty()
    {
        var lookup = PinyinLookup.Instance;
        var (i, f) = lookup.GetInitialsAndFinals("");
        Assert.Empty(i);
        Assert.Empty(f);
    }

    [Fact]
    public void GetInitialsAndFinals_MixedKnownAndPunctuation()
    {
        var lookup = PinyinLookup.Instance;
        const string s = "你,好";
        var expectedI = lookup.GetInitials(s);
        var expectedF = lookup.GetFinals(s);
        var (i, f) = lookup.GetInitialsAndFinals(s);
        Assert.Equal(expectedI, i);
        Assert.Equal(expectedF, f);
    }

    // =========================================================================
    // #7 — PreMergeForModify uses a shared GetWordFinals cache.
    //       The observable behaviour (segment merging) must be unchanged;
    //       we cover the rules that depend on finals directly.
    // =========================================================================

    [Fact]
    public void PreMergeForModify_EmptyInput_ReturnsEmpty()
    {
        var segs = new List<(string word, string pos)>();
        var merged = ToneSandhi.PreMergeForModify(segs);
        Assert.Empty(merged);
    }

    [Fact]
    public void PreMergeForModify_SingleSegment_Unchanged()
    {
        var segs = new List<(string word, string pos)> { ("你好", "l") };
        var merged = ToneSandhi.PreMergeForModify(segs);
        Assert.Single(merged);
        Assert.Equal(("你好", "l"), merged[0]);
    }

    [Fact]
    public void PreMergeForModify_MergeBu_MergesBuAndFollowingWord()
    {
        // 不 + 是 → 不是 (MergeBu rule).
        var segs = new List<(string word, string pos)> { ("不", "d"), ("是", "v") };
        var merged = ToneSandhi.PreMergeForModify(segs);
        Assert.Contains(merged, s => s.word == "不是");
    }

    [Fact]
    public void PreMergeForModify_MergeYi_MergesYiReduplication()
    {
        // 看 + 一 + 看 → 看一看
        var segs = new List<(string word, string pos)> { ("看", "v"), ("一", "m"), ("看", "v") };
        var merged = ToneSandhi.PreMergeForModify(segs);
        Assert.Contains(merged, s => s.word == "看一看");
    }

    [Fact]
    public void PreMergeForModify_IdempotentOnAlreadyMerged()
    {
        // Running twice should be stable.
        var segs = new List<(string word, string pos)> { ("不", "d"), ("是", "v") };
        var once = ToneSandhi.PreMergeForModify(segs);
        var twice = ToneSandhi.PreMergeForModify(once);
        Assert.Equal(once, twice);
    }

    [Fact]
    public void PreMergeForModify_MergeEr_MergesErSuffix()
    {
        // For the MergeEr rule: a lone 儿 following a word should attach.
        var segs = new List<(string word, string pos)> { ("花", "n"), ("儿", "n") };
        var merged = ToneSandhi.PreMergeForModify(segs);
        // Exact shape depends on rule text; we only require the count dropped and 儿 is absorbed.
        Assert.DoesNotContain(merged, s => s.word == "儿");
    }

    // =========================================================================
    // #8 — TextNormalizer trad→simp map is built from a deduplicated literal.
    //       Behaviour: common traditional chars must still map to their
    //       simplified counterparts.
    // =========================================================================

    [Theory]
    [InlineData('國', '国')]
    [InlineData('學', '学')]
    [InlineData('書', '书')]
    [InlineData('車', '车')]
    [InlineData('東', '东')]
    [InlineData('電', '电')]
    [InlineData('話', '话')]
    [InlineData('長', '长')]
    [InlineData('開', '开')]
    [InlineData('關', '关')]
    [InlineData('時', '时')]
    [InlineData('問', '问')]
    [InlineData('題', '题')]
    [InlineData('點', '点')]
    [InlineData('頭', '头')]
    [InlineData('發', '发')]
    [InlineData('現', '现')]
    [InlineData('場', '场')]
    [InlineData('報', '报')]
    [InlineData('導', '导')]
    [InlineData('體', '体')]
    [InlineData('認', '认')]
    [InlineData('識', '识')]
    [InlineData('語', '语')]
    [InlineData('說', '说')]
    [InlineData('還', '还')]
    [InlineData('這', '这')]
    [InlineData('種', '种')]
    [InlineData('從', '从')]
    [InlineData('來', '来')]
    [InlineData('過', '过')]
    [InlineData('後', '后')]
    [InlineData('進', '进')]
    [InlineData('動', '动')]
    [InlineData('員', '员')]
    [InlineData('會', '会')]
    [InlineData('議', '议')]
    [InlineData('經', '经')]
    [InlineData('濟', '济')]
    [InlineData('計', '计')]
    [InlineData('劃', '划')]
    [InlineData('區', '区')]
    [InlineData('別', '别')]
    [InlineData('處', '处')]
    [InlineData('機', '机')]
    [InlineData('構', '构')]
    [InlineData('權', '权')]
    [InlineData('義', '义')]
    [InlineData('務', '务')]
    [InlineData('責', '责')]
    [InlineData('備', '备')]
    [InlineData('準', '准')]
    [InlineData('確', '确')]
    [InlineData('實', '实')]
    [InlineData('際', '际')]
    [InlineData('係', '系')]
    [InlineData('據', '据')]
    [InlineData('擊', '击')]
    [InlineData('選', '选')]
    [InlineData('項', '项')]
    [InlineData('設', '设')]
    [InlineData('節', '节')]
    [InlineData('約', '约')]
    [InlineData('條', '条')]
    [InlineData('記', '记')]
    [InlineData('錄', '录')]
    [InlineData('製', '制')]
    [InlineData('註', '注')]
    [InlineData('冊', '册')]
    [InlineData('號', '号')]
    [InlineData('碼', '码')]
    [InlineData('鏈', '链')]
    [InlineData('築', '筑')]
    [InlineData('線', '线')]
    [InlineData('網', '网')]
    [InlineData('絡', '络')]
    [InlineData('連', '连')]
    [InlineData('結', '结')]
    public void T2S_KnownTraditionalChar_MapsToSimplified(char trad, char simp)
    {
        // Drive the private ConvertTraditionalToSimplified directly so other normalisation
        // (full-width → half-width, date/number handling) doesn't interfere with the assertion.
        var input = new string([trad]);
        var output = ConvertTraditionalToSimplified(input);
        Assert.Contains(simp, output);
    }

    [Fact]
    public void T2S_SimplifiedInput_Unchanged()
    {
        const string s = "中国人";
        Assert.Equal(s, ConvertTraditionalToSimplified(s));
    }

    [Fact]
    public void T2S_EmptyInput_Empty()
    {
        Assert.Equal(string.Empty, ConvertTraditionalToSimplified(""));
    }

    [Fact]
    public void T2S_NonCjkInput_Unchanged()
    {
        const string s = "hello 2026!";
        Assert.Equal(s, ConvertTraditionalToSimplified(s));
    }

    // -- Reflection shim ---------------------------------------------------
    // `ConvertTraditionalToSimplified` is private but is the exact surface
    // affected by the #8 deduplicated-literal change. We call it reflectively
    // rather than going through the full `Normalize` pipeline so that other
    // steps (full-width, number, date) don't mask regressions here.
    private static readonly MethodInfo ConvertT2SMethod =
        typeof(TextNormalizer)
            .GetMethod("ConvertTraditionalToSimplified",
                BindingFlags.Static | BindingFlags.NonPublic)
        ?? throw new InvalidOperationException("ConvertTraditionalToSimplified not found");

    private static string ConvertTraditionalToSimplified(string text)
        => (string)ConvertT2SMethod.Invoke(null, [text])!;
}
