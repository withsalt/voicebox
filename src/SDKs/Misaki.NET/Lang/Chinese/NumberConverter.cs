using System.Text;
using System.Text.RegularExpressions;

namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Converts Arabic numerals in Chinese text to Chinese characters.
/// Implements functionality equivalent to the Python cn2an library's an2cn transform.
/// </summary>
public static class NumberConverter
{
    private static readonly string[] LowDigits =
        { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九" };

    private static readonly string[] DirectDigits =
        { "〇", "一", "二", "三", "四", "五", "六", "七", "八", "九" };

    /// <summary>Unit characters for positional notation (ones, tens, hundreds, thousands).</summary>
    private static readonly string[] Units = { "", "十", "百", "千" };

    /// <summary>
    /// Converts a single digit character to its Chinese equivalent.
    /// Uses 〇 for zero (direct/year style).
    /// </summary>
    public static char DigitToChinese(char digit)
    {
        if (digit >= '0' && digit <= '9')
            return DirectDigits[digit - '0'][0];
        return digit;
    }

    /// <summary>Large unit characters for groups of four digits.</summary>
    private static readonly string[] BigUnits = { "", "万", "亿", "万亿", "兆" };

    // Regex patterns for transform
    private static readonly Regex NumberPattern = new(
        @"-?\d+\.?\d*",
        RegexOptions.Compiled);

    private static readonly Regex DateYearPattern = new(
        @"(\d+)(年)",
        RegexOptions.Compiled);

    private static readonly Regex DateMonthDayPattern = new(
        @"(\d+)(月|日|号)",
        RegexOptions.Compiled);

    private static readonly Regex FractionPattern = new(
        @"(\d+\.?\d*)\s*/\s*(\d+\.?\d*)",
        RegexOptions.Compiled);

    private static readonly Regex PercentagePattern = new(
        @"(\d+\.?\d*)%",
        RegexOptions.Compiled);

    private static readonly Regex CelsiusPattern = new(
        @"(\d+\.?\d*)℃",
        RegexOptions.Compiled);

    private static readonly Regex RangePattern = new(
        @"(?<!\d)(\d+\.?\d*)\s*[-–—~]\s*(\d+\.?\d*)(?!\d*[年月日号])",
        RegexOptions.Compiled);

    /// <summary>
    /// Converts an Arabic numeral string to Chinese characters using lowercase (low) mode.
    /// </summary>
    /// <param name="number">The Arabic numeral string (e.g., "123", "-45.6").</param>
    /// <returns>The Chinese character representation.</returns>
    /// <example>
    /// NumberConverter.An2Cn("12") returns "十二"
    /// NumberConverter.An2Cn("123") returns "一百二十三"
    /// NumberConverter.An2Cn("-5.3") returns "负五点三"
    /// </example>
    public static string An2Cn(string number)
    {
        if (string.IsNullOrWhiteSpace(number))
            return string.Empty;

        number = number.Trim();

        // Handle negative numbers
        if (number.StartsWith('-'))
        {
            return "负" + An2Cn(number[1..]);
        }

        // Handle decimal point
        int dotIndex = number.IndexOf('.');
        if (dotIndex >= 0)
        {
            string intPart = dotIndex == 0 ? "0" : number[..dotIndex];
            string decPart = number[(dotIndex + 1)..];
            string intChinese = ConvertInteger(intPart);
            string decChinese = ConvertDecimal(decPart);
            return intChinese + "点" + decChinese;
        }

        return ConvertInteger(number);
    }

    /// <summary>
    /// Converts digits directly to Chinese characters without positional units.
    /// Used for year numbers (e.g., 2024 → 二〇二四).
    /// </summary>
    /// <param name="number">The Arabic numeral string.</param>
    /// <returns>The Chinese character representation with direct digit mapping.</returns>
    public static string An2CnDirect(string number)
    {
        if (string.IsNullOrWhiteSpace(number))
            return string.Empty;

        var sb = new StringBuilder(number.Length);
        foreach (char c in number.Trim())
        {
            if (c >= '0' && c <= '9')
            {
                sb.Append(DirectDigits[c - '0']);
            }
            else if (c == '.')
            {
                sb.Append('点');
            }
            else if (c == '-')
            {
                sb.Append('负');
            }
        }
        return sb.ToString();
    }

    /// <summary>
    /// Transforms Arabic numerals found in Chinese text to Chinese characters.
    /// Handles special patterns: dates, fractions, percentages, Celsius, and ranges.
    /// This is the main entry point equivalent to cn2an.transform(text, method='an2cn').
    /// </summary>
    /// <param name="text">Input text potentially containing Arabic numerals.</param>
    /// <returns>Text with Arabic numerals replaced by Chinese characters.</returns>
    public static string Transform(string text)
    {
        if (string.IsNullOrEmpty(text))
            return text;

        // Process Celsius first (before general number replacement)
        text = CelsiusPattern.Replace(text, m =>
        {
            string num = An2Cn(m.Groups[1].Value);
            return num + "摄氏度";
        });

        // Process percentages
        text = PercentagePattern.Replace(text, m =>
        {
            string num = An2Cn(m.Groups[1].Value);
            return "百分之" + num;
        });

        // Process fractions (X/Y → Y分之X)
        text = FractionPattern.Replace(text, m =>
        {
            string numerator = An2Cn(m.Groups[1].Value);
            string denominator = An2Cn(m.Groups[2].Value);
            return denominator + "分之" + numerator;
        });

        // Process year dates (2024年 → 二〇二四年)
        text = DateYearPattern.Replace(text, m =>
        {
            string yearDigits = An2CnDirect(m.Groups[1].Value);
            return yearDigits + m.Groups[2].Value;
        });

        // Process month/day dates (12月 → 十二月, 3日 → 三日)
        text = DateMonthDayPattern.Replace(text, m =>
        {
            string num = An2Cn(m.Groups[1].Value);
            return num + m.Groups[2].Value;
        });

        // Process ranges (X-Y → X到Y) - only numeric ranges
        text = RangePattern.Replace(text, m =>
        {
            string left = An2Cn(m.Groups[1].Value);
            string right = An2Cn(m.Groups[2].Value);
            return left + "到" + right;
        });

        // Process remaining standalone numbers
        text = NumberPattern.Replace(text, m => An2Cn(m.Value));

        return text;
    }

    /// <summary>
    /// Converts the integer part of a number to Chinese characters with positional units.
    /// </summary>
    private static string ConvertInteger(string intStr)
    {
        // Remove leading zeros
        intStr = intStr.TrimStart('0');
        if (string.IsNullOrEmpty(intStr))
            return LowDigits[0]; // "零"

        long value = long.Parse(intStr);
        if (value == 0)
            return LowDigits[0];

        string result = ConvertIntegerValue(value);

        // Strip leading 一 from 一十X (e.g., 12 → 十二 not 一十二)
        if (result.StartsWith("一十"))
        {
            result = result[1..];
        }

        return result;
    }

    /// <summary>
    /// Converts a non-negative integer value to Chinese characters.
    /// Handles grouping by 万 (10,000) sections.
    /// </summary>
    private static string ConvertIntegerValue(long value)
    {
        if (value == 0)
            return LowDigits[0];

        var parts = new List<string>();
        int groupIndex = 0;

        while (value > 0)
        {
            int section = (int)(value % 10000);
            if (section != 0)
            {
                string sectionStr = ConvertSection(section);
                if (groupIndex < BigUnits.Length)
                {
                    parts.Add(sectionStr + BigUnits[groupIndex]);
                }
                else
                {
                    parts.Add(sectionStr);
                }
            }
            else if (parts.Count > 0 && !parts[^1].StartsWith(LowDigits[0]))
            {
                // Insert a zero placeholder when a whole section is zero
                // but only if the previous part doesn't already start with zero
                parts.Add(LowDigits[0]);
            }

            value /= 10000;
            groupIndex++;
        }

        parts.Reverse();

        var sb = new StringBuilder();
        for (int i = 0; i < parts.Count; i++)
        {
            sb.Append(parts[i]);
        }

        // Clean up multiple consecutive zeros
        string result = sb.ToString();
        while (result.Contains("零零"))
        {
            result = result.Replace("零零", "零");
        }

        // Remove trailing zero
        if (result.EndsWith("零") && result.Length > 1)
        {
            result = result[..^1];
        }

        return result;
    }

    /// <summary>
    /// Converts a 4-digit section (0-9999) to Chinese characters with units.
    /// </summary>
    private static string ConvertSection(int section)
    {
        if (section == 0)
            return string.Empty;

        var sb = new StringBuilder();
        int[] digits = new int[4];
        digits[0] = section / 1000;       // thousands
        digits[1] = (section / 100) % 10; // hundreds
        digits[2] = (section / 10) % 10;  // tens
        digits[3] = section % 10;         // ones

        bool needZero = false;

        for (int i = 0; i < 4; i++)
        {
            if (digits[i] != 0)
            {
                if (needZero)
                {
                    sb.Append(LowDigits[0]);
                    needZero = false;
                }
                sb.Append(LowDigits[digits[i]]);
                sb.Append(Units[3 - i]);
            }
            else
            {
                // Only set needZero if we've already written something
                if (sb.Length > 0)
                {
                    needZero = true;
                }
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Converts the decimal part of a number to Chinese characters (digit by digit).
    /// </summary>
    private static string ConvertDecimal(string decStr)
    {
        var sb = new StringBuilder(decStr.Length);
        foreach (char c in decStr)
        {
            if (c >= '0' && c <= '9')
            {
                sb.Append(LowDigits[c - '0']);
            }
        }
        return sb.ToString();
    }
}
