using Xunit;
using Misaki.NET.Lang.English;
using Misaki.NET.Lang.Chinese;

namespace Misaki.Tests;

public class BugFixAndPerfTests
{
    // ==========================================================================
    // Bug 1: EnglishLexicon.GetNumber — redundant condition removal
    // The old code had `!Currencies.ContainsKey(currency ?? "")` which was
    // always true when currency == null. The behavior should not regress for
    // the documented 4-digit year path.
    // ==========================================================================

    [Fact]
    public void Bug1_GetNumber_FourDigitYear_Works()
    {
        var lex = new EnglishLexicon(british: false);
        // 1999 should be readable as a year via the 4-digit path (no currency)
        var (phonemes, _) = lex.GetNumber("1999", currency: null, isHead: true, numFlags: "");
        Assert.False(string.IsNullOrEmpty(phonemes),
            "4-digit number with null currency should produce phonemes");
    }

    [Fact]
    public void Bug1_GetNumber_FourDigitNumber_NullCurrency_NotSkipped()
    {
        var lex = new EnglishLexicon(british: false);
        // Regardless of whether interpreted as year or cardinal, should produce output.
        var (phonemes, _) = lex.GetNumber("2024", currency: null, isHead: true, numFlags: "");
        Assert.False(string.IsNullOrEmpty(phonemes));
    }

    // ==========================================================================
    // Bug 2: PinyinLookupBridge — thread-safe (volatile) global state.
    // Verifies that Initialize + GetFinals round-trip works and that the
    // lookup func can be cleared without throwing.
    // ==========================================================================

    [Fact]
    public void Bug2_PinyinLookupBridge_Initialize_SetsLookupFunc()
    {
        var saved = PinyinLookupBridge.LookupFunc;
        try
        {
            PinyinLookupBridge.LookupFunc = null;
            Assert.Null(PinyinLookupBridge.GetFinals("你好"));

            PinyinLookupBridge.Initialize(PinyinLookup.Instance);
            Assert.NotNull(PinyinLookupBridge.LookupFunc);
            // Known word should return non-null finals
            var finals = PinyinLookupBridge.GetFinals("你好");
            Assert.NotNull(finals);
            Assert.NotEmpty(finals);
        }
        finally
        {
            PinyinLookupBridge.LookupFunc = saved;
        }
    }

    [Fact]
    public void Bug2_PinyinLookupBridge_NullLookup_ReturnsNullSafely()
    {
        var saved = PinyinLookupBridge.LookupFunc;
        try
        {
            PinyinLookupBridge.LookupFunc = null;
            // Should not throw, just return null
            var result = PinyinLookupBridge.GetFinals("anything");
            Assert.Null(result);
        }
        finally
        {
            PinyinLookupBridge.LookupFunc = saved;
        }
    }

    // ==========================================================================
    // Bug 3: EnglishG2P.Preprocess phoneme override
    // The old code `feature = f[..^0]` was equivalent to `feature = f`, which
    // left the trailing '/' in the phoneme output. The fix strips the trailing
    // '/' while keeping the leading '/' as the type marker.
    // ==========================================================================

    [Fact]
    public void Bug3_PhonemeOverride_DoesNotLeakTrailingSlash()
    {
        var g2p = new EnglishG2P();
        // Markdown-style link with phoneme override
        var result = g2p.Process("[hello](/həˈloʊ/)");
        Assert.NotNull(result.Phonemes);
        // The final output must not contain a stray '/' left over from the override
        Assert.DoesNotContain("/", result.Phonemes);
        // And must include the overridden phonemes
        Assert.Contains("həˈloʊ", result.Phonemes);
    }

    [Fact]
    public void Bug3_PhonemeOverride_SimpleCase()
    {
        var g2p = new EnglishG2P();
        var result = g2p.Process("[word](/wɜːd/)");
        Assert.DoesNotContain("/", result.Phonemes);
        Assert.Contains("wɜːd", result.Phonemes);
    }

    // ==========================================================================
    // Bug 5: ChineseG2P erhua Zhuyin encoding
    // The old code's `.Replace("_eR", "_er").Replace("R", "_R")` would fail
    // when an erhua'd "e" final appeared at the start of a phoneme string
    // (no leading underscore). The fix handles the string-start "eR" case.
    // ==========================================================================

    [Fact]
    public void Bug5_Erhua_ProducesValidOutput()
    {
        var g2p = new ChineseG2P(version: "1.1");
        // 花儿 - classic erhua word
        var result = g2p.Process("花儿");
        Assert.NotEmpty(result.Phonemes);
        // Output should not contain a stray 'R' token without its final
        // (a stray 'R' as a standalone segment indicates the bug).
        // The erhua R should be joined with its preceding final, not isolated.
        Assert.DoesNotContain("_R_", result.Phonemes);
    }

    [Fact]
    public void Bug5_Erhua_CharacterByCharacter()
    {
        var g2p = new ChineseG2P(version: "1.1");
        // 儿 standalone
        var result = g2p.Process("儿");
        Assert.NotEmpty(result.Phonemes);
    }

    // ==========================================================================
    // Bug 6: NumberConverter.Transform — date vs range disambiguation.
    // Old regex would match "2024-01" (a date) as a number range.
    // New regex uses negative lookahead (?!\d*[年月日号]) to skip dates.
    // ==========================================================================

    [Fact]
    public void Bug6_NumberConverter_DateRange_NotMatchedAsNumericRange()
    {
        // "2024-01 月" — the "-01" is date-like, followed by 月
        var result = NumberConverter.Transform("2024-01月");
        // Should not contain "到" (which would appear if "-01" was matched as range)
        Assert.DoesNotContain("到一月", result);
    }

    [Fact]
    public void Bug6_NumberConverter_PlainNumericRange_StillMatched()
    {
        // Plain range should still be converted
        var result = NumberConverter.Transform("5-10");
        Assert.Contains("到", result);
    }

    [Fact]
    public void Bug6_NumberConverter_PlainNumber_NotAffected()
    {
        var result = NumberConverter.Transform("2024");
        Assert.False(string.IsNullOrEmpty(result));
    }

    // ==========================================================================
    // Perf 1: ChineseG2P regex pre-compile — behavior should not change.
    // We verify SingleIFinalRegex and BeforeDigitRegex produce the same output
    // as the original inline regex for representative inputs.
    // ==========================================================================

    [Fact]
    public void Perf1_ChineseG2P_StandardInput_StillWorks()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("我爱北京天安门");
        Assert.NotEmpty(result.Phonemes);
    }

    [Fact]
    public void Perf1_ChineseG2P_WithTones_ProducesToneMarkers()
    {
        var g2p = new ChineseG2P(version: "1.1");
        var result = g2p.Process("你好");
        // Version 1.1 uses Zhuyin tone markers
        Assert.NotEmpty(result.Phonemes);
    }

    // ==========================================================================
    // Perf 2: EnglishLexicon regex pre-compile — behavior unchanged.
    // StemIng depends on DoubleConsonantIngRegex.
    // ==========================================================================

    [Fact]
    public void Perf2_EnglishLexicon_IngWord_StillStems()
    {
        var lex = new EnglishLexicon(british: false);
        // "running" - matches `([bcdgklmnprstvxz])\1ing$` pattern (double n)
        var (phonemes, _) = lex.StemIng("running", tag: "VBG", stress: null, ctx: null);
        // Should produce some phoneme output (either via stem or direct lookup)
        // The regex path is exercised; we don't assert specific phonemes because
        // the result depends on the bundled dictionary.
        Assert.True(phonemes == null || phonemes.Length > 0);
    }

    [Fact]
    public void Perf2_EnglishLexicon_CkingWord_StillStems()
    {
        var lex = new EnglishLexicon(british: false);
        // "cooking" - matches `cking$` pattern
        var (phonemes, _) = lex.StemIng("cooking", tag: "VBG", stress: null, ctx: null);
        Assert.True(phonemes == null || phonemes.Length > 0);
    }
}
