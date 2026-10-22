using Xunit;
using Misaki.NET.Lang.Chinese;

namespace Misaki.Tests;

public class ChineseG2PTests
{
    private readonly ChineseG2P _g2p = new(version: "1.1");

    [Fact]
    public void Process_EmptyString_ReturnsEmpty()
    {
        var result = _g2p.Process("");
        Assert.Equal("", result.Phonemes);
    }

    [Fact]
    public void Process_NullInput_ReturnsEmpty()
    {
        var result = _g2p.Process(null!);
        Assert.Equal("", result.Phonemes);
    }

    [Fact]
    public void Process_WhitespaceOnly_ReturnsEmpty()
    {
        var result = _g2p.Process("   ");
        Assert.Equal("", result.Phonemes);
    }

    [Fact]
    public void Process_BasicChinese_ReturnsPhonemes()
    {
        var result = _g2p.Process("你好");
        Assert.NotEmpty(result.Phonemes);
        // Should contain Zhuyin characters
        Assert.Contains("ㄋ", result.Phonemes);
    }

    [Fact]
    public void Process_BasicSentence_ReturnsPhonemes()
    {
        var result = _g2p.Process("今天天气真好");
        Assert.NotEmpty(result.Phonemes);
    }

    [Fact]
    public void Process_ChinesePunctuation_MapsToAscii()
    {
        var result = _g2p.Process("你好！");
        // Chinese punctuation should be mapped to ASCII equivalents
        Assert.Contains("!", result.Phonemes);
    }

    [Fact]
    public void Process_MixedChineseEnglish_HandlesEnglish()
    {
        var g2p = new ChineseG2P(version: "1.1", enCallable: en => $"[{en}]");
        var result = g2p.Process("我喜欢Python编程");
        Assert.Contains("[Python]", result.Phonemes);
    }

    [Fact]
    public void Process_NumbersInText_ConvertsToChineseFirst()
    {
        // Numbers should be converted to Chinese before phoneme processing
        var result = _g2p.Process("我有三个苹果");
        Assert.NotEmpty(result.Phonemes);
        // Should contain Zhuyin for 三 (ㄙㄢ)
        Assert.Contains("ㄙ", result.Phonemes);
    }

    [Fact]
    public void Process_LongText_DoesNotThrow()
    {
        var longText = string.Concat(Enumerable.Repeat("今天天气真好，我们一起去公园散步吧。", 10));
        var result = _g2p.Process(longText);
        Assert.NotEmpty(result.Phonemes);
    }

    // Regression: y/w zero-consonant syllables previously mis-extracted y/w as
    // initials, producing unknown-phoneme placeholders for yun/yao/wo/wen
    // and similar (e.g. 运 → ❓文4 instead of 云4).
    // ZH_MAP picks the Chinese-character glyphs as Zhuyin shorthands here:
    //   vn → 云, iao → 要, uo → 我, ve → 月, vn1 → ㄐ云, etc.
    [Theory]
    [InlineData("运行", "云", "❓")]       // yun4 xing2: must contain 云, never the unk marker
    [InlineData("我要", "要", "❓")]       // yao4: must contain 要
    [InlineData("我们", "我", "❓")]       // wo3: must contain 我
    [InlineData("觉得", "月", "❓")]       // jue2: must contain 月 (via ve)
    [InlineData("军人", "云", "❓")]       // jun1: must contain 云 (via vn)
    [InlineData("全国", "元", "❓")]       // quan2: must contain 元 (via van)
    [InlineData("学生", "月", "❓")]       // xue2: must contain 月
    public void Process_ZeroConsonantCases_DoesNotProduceUnk(
        string text, string mustContain, string mustNotContain)
    {
        var result = _g2p.Process(text);
        Assert.Contains(mustContain, result.Phonemes);
        Assert.DoesNotContain(mustNotContain, result.Phonemes);
    }
}

public class ChineseNumberConverterTests
{
    [Theory]
    [InlineData("0", "零")]
    [InlineData("1", "一")]
    [InlineData("10", "十")]
    [InlineData("11", "十一")]
    [InlineData("20", "二十")]
    [InlineData("100", "一百")]
    [InlineData("123", "一百二十三")]
    [InlineData("1000", "一千")]
    [InlineData("10000", "一万")]
    public void An2Cn_BasicNumbers_ConvertsCorrectly(string input, string expected)
    {
        var result = NumberConverter.An2Cn(input);
        Assert.Equal(expected, result);
    }

    [Fact]
    public void An2Cn_NegativeNumber_HasPrefix()
    {
        var result = NumberConverter.An2Cn("-5");
        Assert.StartsWith("负", result);
    }

    [Fact]
    public void Transform_DatePattern_ConvertsYearDigitByDigit()
    {
        var result = NumberConverter.Transform("2024年");
        Assert.Contains("二", result);
        Assert.Contains("〇", result);
        Assert.Contains("年", result);
    }

    [Fact]
    public void Transform_Percentage_ConvertsCorrectly()
    {
        var result = NumberConverter.Transform("50%");
        Assert.Contains("百分之", result);
    }

    [Fact]
    public void DigitToChinese_AllDigits_ConvertsCorrectly()
    {
        Assert.Equal('〇', NumberConverter.DigitToChinese('0'));
        Assert.Equal('一', NumberConverter.DigitToChinese('1'));
        Assert.Equal('九', NumberConverter.DigitToChinese('9'));
    }
}

public class ChineseTextNormalizerTests
{
    [Fact]
    public void Normalize_EmptyString_ReturnsEmpty()
    {
        Assert.Equal("", TextNormalizer.Normalize(""));
    }

    [Fact]
    public void Normalize_FullWidthChars_ConvertsToHalfWidth()
    {
        // Test just the full-width to half-width conversion step
        var result = TextNormalizer.NormalizeSentence("ＸＹＺ");
        Assert.Equal("XYZ", result);
    }

    [Fact]
    public void Normalize_Temperature_ConvertsCorrectly()
    {
        var result = TextNormalizer.NormalizeSentence("温度是25°C");
        Assert.Contains("二十五", result);
        Assert.Contains("度", result);
    }

    [Fact]
    public void Normalize_Time_ConvertsCorrectly()
    {
        var result = TextNormalizer.NormalizeSentence("现在是12:30");
        Assert.Contains("十二", result);
        Assert.Contains("半", result);
    }

    [Fact]
    public void Normalize_Fraction_ConvertsCorrectly()
    {
        var result = TextNormalizer.NormalizeSentence("占比为3/4");
        Assert.Contains("四分之三", result);
    }
}

public class PinyinToIpaTests
{
    [Theory]
    [InlineData("ma1", "˥")]
    [InlineData("ma2", "˧˥")]
    [InlineData("ma3", "˧˩˧")]
    [InlineData("ma4", "˥˩")]
    [InlineData("ma5", "")]
    public void Convert_ToneMarkers_IncludedCorrectly(string pinyin, string expectedTone)
    {
        var result = PinyinToIpa.Convert(pinyin);
        if (!string.IsNullOrEmpty(expectedTone))
            Assert.Contains(expectedTone, result);
        else
            Assert.DoesNotContain("˥", result);
    }

    [Fact]
    public void Convert_EmptyString_ReturnsEmpty()
    {
        Assert.Equal("", PinyinToIpa.Convert(""));
    }

    [Fact]
    public void Retone_ThirdTone_ReturnsArrow()
    {
        var ipa = "ma˧˩˧";
        var result = PinyinToIpa.Retone(ipa);
        Assert.Contains("↓", result);
    }

    [Fact]
    public void Retone_SecondTone_ReturnsArrow()
    {
        var ipa = "ma˧˥";
        var result = PinyinToIpa.Retone(ipa);
        Assert.Contains("↗", result);
    }
}

public class PinyinLookupTests
{
    [Fact]
    public void GetTone3_SingleChar_ReturnsCorrectPinyin()
    {
        var lookup = PinyinLookup.Instance;
        var result = lookup.GetTone3("你");
        Assert.Single(result);
        Assert.Equal("ni3", result[0]);
    }

    [Fact]
    public void GetTone3_MultiChar_ReturnsPinyinForEach()
    {
        var lookup = PinyinLookup.Instance;
        var result = lookup.GetTone3("你好");
        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void GetInitials_BasicWord_ReturnsCorrectInitials()
    {
        var lookup = PinyinLookup.Instance;
        var result = lookup.GetInitials("中");
        Assert.Single(result);
        Assert.Equal("zh", result[0]);
    }

    [Fact]
    public void GetFinals_BasicWord_ReturnsCorrectFinals()
    {
        var lookup = PinyinLookup.Instance;
        var result = lookup.GetFinals("中");
        Assert.Single(result);
        Assert.StartsWith("ong", result[0]);
    }

    [Fact]
    public void IsPhrase_KnownPhrase_ReturnsTrue()
    {
        var lookup = PinyinLookup.Instance;
        // Common phrases should be in the dictionary
        Assert.True(lookup.IsPhrase("中国"));
    }

    [Fact]
    public void GetTone3_UnknownChar_ReturnsEmpty()
    {
        var lookup = PinyinLookup.Instance;
        var result = lookup.GetTone3("☺");
        Assert.Single(result);
        Assert.Equal("", result[0]);
    }

    // Zero-consonant restoration (pypinyin strict mode parity).
    // y/w are NOT initials — they mark zero-consonant syllables that must be
    // restored to their underlying form before initial/final extraction.
    [Theory]
    [InlineData("要", "", "iao4")]    // yao4 → iao4
    [InlineData("运", "", "vn4")]     // yun4 → vn4 (via ün)
    [InlineData("月", "", "ve4")]     // yue4 → ve4
    [InlineData("元", "", "van2")]    // yuan2 → van2
    [InlineData("于", "", "v2")]      // yu2 → v2
    [InlineData("阳", "", "iang2")]   // yang2 → iang2
    [InlineData("用", "", "iong4")]   // yong4 → iong4
    [InlineData("又", "", "iou4")]    // you4 → iou4
    [InlineData("衣", "", "i1")]      // yi1 → i1
    [InlineData("应", "", "ing1")]    // ying1 → ing1
    [InlineData("我", "", "uo3")]     // wo3 → uo3
    [InlineData("文", "", "uen2")]    // wen2 → uen2
    [InlineData("外", "", "uai4")]    // wai4 → uai4
    [InlineData("万", "", "uan4")]    // wan4 → uan4
    [InlineData("为", "", "uei4")]    // wei4 → uei4
    [InlineData("五", "", "u3")]      // wu3 → u3
    [InlineData("翁", "", "ueng1")]   // weng1 → ueng1
    // j/q/x + u → j/q/x + ü
    [InlineData("居", "j", "v1")]     // ju1 → j + v1
    [InlineData("觉", "j", "ve2")]    // jue2 → j + ve2
    [InlineData("卷", "j", "van3")]   // juan3 → j + van3
    [InlineData("军", "j", "vn1")]    // jun1 → j + vn1
    [InlineData("区", "q", "v1")]
    [InlineData("却", "q", "ve4")]
    [InlineData("全", "q", "van2")]
    [InlineData("群", "q", "vn2")]
    [InlineData("虚", "x", "v1")]
    [InlineData("学", "x", "ve2")]
    [InlineData("选", "x", "van3")]
    [InlineData("训", "x", "vn4")]
    // Control cases: real initials should not be affected.
    [InlineData("时", "sh", "i2")]
    [InlineData("候", "h", "ou4")]
    public void GetInitialsAndFinals_ZeroConsonantAndUv_MatchesPypinyinStrict(
        string ch, string expectedInitial, string expectedFinal)
    {
        var lookup = PinyinLookup.Instance;
        var (initials, finals) = lookup.GetInitialsAndFinals(ch);
        Assert.Single(initials);
        Assert.Single(finals);
        Assert.Equal(expectedInitial, initials[0]);
        Assert.Equal(expectedFinal, finals[0]);
    }
}

public class ToneSandhiTests
{
    [Fact]
    public void ModifiedTone_BuBeforeTone4_ChangesToTone2()
    {
        // 不怕: 不 before tone 4 should become tone 2
        var finals = new List<string> { "u4", "a4" };
        var result = ToneSandhi.ModifiedTone("不怕", "v", finals);
        Assert.Equal("u2", result[0]); // 不 changes to tone 2
    }

    [Fact]
    public void ModifiedTone_YiBeforeTone4_ChangesToTone2()
    {
        // 一段: 一 before tone 4 should become tone 2
        var finals = new List<string> { "i1", "uan4" };
        var result = ToneSandhi.ModifiedTone("一段", "m", finals);
        Assert.Equal("i2", result[0]);
    }

    [Fact]
    public void ModifiedTone_ThreeSandhi_FirstChangesToTone2()
    {
        // Two consecutive tone 3: first becomes tone 2
        var finals = new List<string> { "i3", "ao3" };
        var result = ToneSandhi.ModifiedTone("你好", "n", finals);
        Assert.Equal("i2", result[0]);
    }

    [Fact]
    public void PreMergeForModify_MergesBu()
    {
        var segments = new List<(string word, string pos)>
        {
            ("不", "d"), ("好", "a")
        };
        var result = ToneSandhi.PreMergeForModify(segments);
        Assert.Contains(result, s => s.word == "不好");
    }
}
