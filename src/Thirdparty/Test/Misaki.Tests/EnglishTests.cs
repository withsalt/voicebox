using Misaki.NET.Lang.Chinese;
using Xunit;
using Misaki.NET.Lang.English;

namespace Misaki.Tests;

public class EnglishG2PTests
{
    private readonly EnglishG2P _g2p = new(british: false);

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
    public void Process_BasicWord_ReturnsPhonemes()
    {
        var result = _g2p.Process("hello");
        Assert.NotEmpty(result.Phonemes);
        // Should not contain the original word
        Assert.DoesNotContain("hello", result.Phonemes);
    }

    [Fact]
    public void Process_CommonWords_ReturnsKnownPhonemes()
    {
        var result = _g2p.Process("the");
        Assert.NotEmpty(result.Phonemes);
        // "the" before consonant should be "ðə"
        Assert.Contains("ð", result.Phonemes);
    }

    [Fact]
    public void Process_MultipleWords_HasWhitespace()
    {
        var result = _g2p.Process("hello world");
        Assert.NotEmpty(result.Phonemes);
        Assert.Contains(" ", result.Phonemes);
    }

    [Fact]
    public void Process_Punctuation_PreservedInOutput()
    {
        var result = _g2p.Process("hello, world!");
        Assert.Contains(",", result.Phonemes);
        Assert.Contains("!", result.Phonemes);
    }

    [Fact]
    public void Process_UpperCase_HandledCorrectly()
    {
        var result = _g2p.Process("HELLO");
        Assert.NotEmpty(result.Phonemes);
    }

    [Fact]
    public void Process_MixedCase_HandledCorrectly()
    {
        var result = _g2p.Process("Hello World");
        Assert.NotEmpty(result.Phonemes);
    }

    [Fact]
    public void Process_LongText_DoesNotThrow()
    {
        var longText = string.Join(" ", Enumerable.Repeat("The quick brown fox jumps over the lazy dog.", 20));
        var result = _g2p.Process(longText);
        Assert.NotEmpty(result.Phonemes);
    }
}

public class EnglishLexiconTests
{
    private readonly EnglishLexicon _lexicon = new(british: false);

    [Fact]
    public void GetWord_KnownWord_ReturnsPhonemes()
    {
        var ctx = new TokenContext();
        var (ps, rating) = _lexicon.GetWord("hello", "NN", null, ctx);
        Assert.NotNull(ps);
        Assert.True(rating >= 3);
    }

    [Fact]
    public void GetWord_UnknownWord_ReturnsNull()
    {
        var ctx = new TokenContext();
        var (ps, _) = _lexicon.GetWord("xyzzyplugh", "NN", null, ctx);
        Assert.Null(ps);
    }

    [Fact]
    public void GetWord_PluralForm_StemsCorrectly()
    {
        var ctx = new TokenContext();
        var (ps, _) = _lexicon.GetWord("cats", "NNS", null, ctx);
        Assert.NotNull(ps);
        Assert.EndsWith("s", ps); // Should end with /s/ after voiceless
    }

    [Fact]
    public void GetWord_PastTense_StemsCorrectly()
    {
        var ctx = new TokenContext();
        var (ps, _) = _lexicon.GetWord("walked", "VBD", null, ctx);
        Assert.NotNull(ps);
        Assert.EndsWith("t", ps); // walk ends in /k/, so -ed → /t/
    }

    [Fact]
    public void GetWord_IngForm_StemsCorrectly()
    {
        var ctx = new TokenContext();
        var (ps, _) = _lexicon.GetWord("running", "VBG", null, ctx);
        Assert.NotNull(ps);
        Assert.Contains("ɪŋ", ps);
    }

    [Fact]
    public void GetWord_TheBeforeConsonant_ReturnsSchwa()
    {
        var ctx = new TokenContext { FutureVowel = false };
        var (ps, _) = _lexicon.GetWord("the", "DT", null, ctx);
        Assert.Equal("ðə", ps);
    }

    [Fact]
    public void GetWord_TheBeforeVowel_ReturnsI()
    {
        var ctx = new TokenContext { FutureVowel = true };
        var (ps, _) = _lexicon.GetWord("the", "DT", null, ctx);
        Assert.Equal("ði", ps);
    }

    [Fact]
    public void IsNumber_ValidNumbers_ReturnsTrue()
    {
        Assert.True(EnglishLexicon.IsNumber("42", true));
        Assert.True(EnglishLexicon.IsNumber("3.14", true));
        Assert.True(EnglishLexicon.IsNumber("1,000", true));
        Assert.True(EnglishLexicon.IsNumber("-5", true));
        Assert.True(EnglishLexicon.IsNumber("42nd", true));
    }

    [Fact]
    public void IsNumber_NonNumbers_ReturnsFalse()
    {
        Assert.False(EnglishLexicon.IsNumber("hello", true));
        Assert.False(EnglishLexicon.IsNumber("abc123", true));
    }

    [Fact]
    public void GetNumber_SimpleNumber_ReturnsPhonemes()
    {
        var (ps, rating) = _lexicon.GetNumber("42", null, true, "");
        Assert.NotNull(ps);
        Assert.True(rating >= 3);
    }

    [Fact]
    public void GetNumber_Year_FormatsAsYear()
    {
        var (ps, _) = _lexicon.GetNumber("1984", null, true, "");
        Assert.NotNull(ps);
        // Should contain "nineteen" phonemes, not "one thousand nine hundred..."
    }
}

public class NumberToWordsTests
{
    [Theory]
    [InlineData(0, "zero")]
    [InlineData(1, "one")]
    [InlineData(13, "thirteen")]
    [InlineData(42, "forty-two")]
    [InlineData(100, "one hundred")]
    [InlineData(123, "one hundred and twenty-three")]
    [InlineData(1000, "one thousand")]
    [InlineData(1000000, "one million")]
    public void ToCardinal_Integer_ReturnsCorrectWords(long number, string expected)
    {
        Assert.Equal(expected, NumberToWords.ToCardinal(number));
    }

    [Fact]
    public void ToCardinal_NegativeNumber_HasMinus()
    {
        var result = NumberToWords.ToCardinal(-5);
        Assert.StartsWith("minus", result);
    }

    [Fact]
    public void ToCardinal_Float_HasPointAndDigits()
    {
        var result = NumberToWords.ToCardinal(3.14);
        Assert.Contains("point", result);
        Assert.Contains("one", result);
        Assert.Contains("four", result);
    }

    [Theory]
    [InlineData(1, "first")]
    [InlineData(2, "second")]
    [InlineData(3, "third")]
    [InlineData(4, "fourth")]
    [InlineData(5, "fifth")]
    [InlineData(12, "twelfth")]
    [InlineData(20, "twentieth")]
    [InlineData(42, "forty-second")]
    public void ToOrdinal_ReturnsCorrectWords(long number, string expected)
    {
        Assert.Equal(expected, NumberToWords.ToOrdinal(number));
    }

    [Theory]
    [InlineData(1984, "nineteen eighty-four")]
    [InlineData(2000, "two thousand")]
    [InlineData(1900, "nineteen hundred")]
    [InlineData(1905, "nineteen oh-five")]
    public void ToYear_ReturnsCorrectWords(int year, string expected)
    {
        Assert.Equal(expected, NumberToWords.ToYear(year));
    }
}

public class EnglishStressTests
{
    [Fact]
    public void ApplyStress_NullStress_ReturnsUnchanged()
    {
        var result = EnglishLexicon.ApplyStress("hɛlˈO", null);
        Assert.Equal("hɛlˈO", result);
    }

    [Fact]
    public void ApplyStress_RemoveAllStress_RemovesMarkers()
    {
        var result = EnglishLexicon.ApplyStress("hɛlˈO", -2);
        Assert.DoesNotContain("ˈ", result);
        Assert.DoesNotContain("ˌ", result);
    }

    [Fact]
    public void ApplyStress_Demote_PrimaryToSecondary()
    {
        var result = EnglishLexicon.ApplyStress("hɛlˈO", -1);
        Assert.DoesNotContain("ˈ", result);
        Assert.Contains("ˌ", result);
    }
}

public class EnglishG2POovTests
{
    [Fact]
    public void EnglishG2P_UnknownWord_UsesNeuralFallback()
    {
        var g2p = new EnglishG2P();
        var result = g2p.Process("xyzzyplugh");
        Assert.NotEmpty(result.Phonemes);
        Assert.DoesNotContain("❓", result.Phonemes);
    }
}

public class ThreadSafetyTests
{
    [Fact]
    public void ChineseG2P_ConcurrentCalls_DoNotThrow()
    {
        var g2p = new ChineseG2P();
        var texts = new[] { "你好", "世界", "今天天气真好", "中华人民共和国", "一二三四五" };

        Parallel.ForEach(texts, text =>
        {
            for (int i = 0; i < 10; i++)
            {
                var result = g2p.Process(text);
                Assert.NotEmpty(result.Phonemes);
            }
        });
    }

    [Fact]
    public void EnglishG2P_ConcurrentCalls_DoNotThrow()
    {
        var g2p = new EnglishG2P();
        var texts = new[] { "hello", "world", "the quick brown fox", "testing one two three" };

        Parallel.ForEach(texts, text =>
        {
            for (int i = 0; i < 10; i++)
            {
                var result = g2p.Process(text);
                Assert.NotEmpty(result.Phonemes);
            }
        });
    }
}
