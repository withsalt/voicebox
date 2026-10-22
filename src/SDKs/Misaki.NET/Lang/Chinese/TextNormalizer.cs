using System.Text;
using System.Text.RegularExpressions;

namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Normalizes Non-Standard-Words (NSW) in Chinese text for TTS.
/// Handles dates, times, numbers, measurements, and special characters.
/// </summary>
public static partial class TextNormalizer
{
    #region Regex Patterns

    private static readonly Regex ReDateChinese = new(
        @"(\d{2,4})年((\d{1,2})月)?((\d{1,2})[日号])?",
        RegexOptions.Compiled);

    private static readonly Regex ReDateSlash = new(
        @"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})",
        RegexOptions.Compiled);

    private static readonly Regex ReTime = new(
        @"(\d{1,2}):(\d{1,2})(:(\d{1,2}))?",
        RegexOptions.Compiled);

    private static readonly Regex ReTemperature = new(
        @"(-?\d+\.?\d*)(°C|℃|度|°F)",
        RegexOptions.Compiled);

    private static readonly Regex ReFraction = new(
        @"(-?\d+)/(\d+)",
        RegexOptions.Compiled);

    private static readonly Regex RePercentage = new(
        @"(\d+\.?\d*)%",
        RegexOptions.Compiled);

    private static readonly Regex ReMobilePhone = new(
        @"(?<!\d)(1[3-9]\d{9})(?!\d)",
        RegexOptions.Compiled);

    private static readonly Regex ReTelephone = new(
        @"(?<!\d)(0\d{2,3}-?\d{7,8})(?!\d)",
        RegexOptions.Compiled);

    private static readonly Regex ReRange = new(
        @"(\d+\.?\d*)[~\-－—](\d+\.?\d*)",
        RegexOptions.Compiled);

    private static readonly Regex ReNumber = new(
        @"(-?\d+)(\.\d+)?",
        RegexOptions.Compiled);

    private static readonly Regex ReSentenceSplit = new(
        @"([。！？；\n]+)",
        RegexOptions.Compiled);

    #endregion

    #region Measurement Units

    private static readonly Dictionary<string, string> MeasureDict = new()
    {
        ["cm²"] = "平方厘米", ["m²"] = "平方米", ["km²"] = "平方公里",
        ["cm"] = "厘米", ["mm"] = "毫米", ["km"] = "千米",
        ["m³"] = "立方米", ["cm³"] = "立方厘米",
        ["kg"] = "千克", ["mg"] = "毫克", ["g"] = "克",
        ["ml"] = "毫升", ["dL"] = "分升", ["L"] = "升",
        ["kHz"] = "千赫兹", ["Hz"] = "赫兹", ["MHz"] = "兆赫兹", ["GHz"] = "吉赫兹",
        ["km/h"] = "千米每小时", ["m/s"] = "米每秒",
        ["kW"] = "千瓦", ["MW"] = "兆瓦", ["W"] = "瓦",
        ["kV"] = "千伏", ["mV"] = "毫伏", ["V"] = "伏特",
        ["mA"] = "毫安", ["A"] = "安培",
        ["kPa"] = "千帕", ["Pa"] = "帕斯卡",
        ["GB"] = "吉字节", ["MB"] = "兆字节", ["KB"] = "千字节", ["TB"] = "太字节",
    };

    #endregion

    #region Greek Letters

    private static readonly Dictionary<string, string> GreekLetters = new()
    {
        ["α"] = "阿尔法", ["β"] = "贝塔", ["γ"] = "伽马", ["δ"] = "德尔塔",
        ["ε"] = "艾普西龙", ["ζ"] = "泽塔", ["η"] = "伊塔", ["θ"] = "西塔",
        ["ι"] = "约塔", ["κ"] = "卡帕", ["λ"] = "兰姆达", ["μ"] = "缪",
        ["ν"] = "纽", ["ξ"] = "克西", ["ο"] = "奥密克戎", ["π"] = "派",
        ["ρ"] = "柔", ["σ"] = "西格马", ["τ"] = "陶", ["υ"] = "宇普西龙",
        ["φ"] = "斐", ["χ"] = "希", ["ψ"] = "普西", ["ω"] = "欧米伽",
        ["Α"] = "阿尔法", ["Β"] = "贝塔", ["Γ"] = "伽马", ["Δ"] = "德尔塔",
        ["Ε"] = "艾普西龙", ["Ζ"] = "泽塔", ["Η"] = "伊塔", ["Θ"] = "西塔",
        ["Ι"] = "约塔", ["Κ"] = "卡帕", ["Λ"] = "兰姆达", ["Μ"] = "缪",
        ["Ν"] = "纽", ["Ξ"] = "克西", ["Ο"] = "奥密克戎", ["Π"] = "派",
        ["Ρ"] = "柔", ["Σ"] = "西格马", ["Τ"] = "陶", ["Υ"] = "宇普西龙",
        ["Φ"] = "斐", ["Χ"] = "希", ["Ψ"] = "普西", ["Ω"] = "欧米伽",
    };

    #endregion

    #region Full-width to Half-width

    private static readonly Dictionary<char, char> FullToHalf;

    static TextNormalizer()
    {
        FullToHalf = new Dictionary<char, char>();
        // ASCII letters full-width to half-width
        for (int i = 0; i < 26; i++)
        {
            FullToHalf[(char)(0xFF21 + i)] = (char)('A' + i); // Ａ-Ｚ
            FullToHalf[(char)(0xFF41 + i)] = (char)('a' + i); // ａ-ｚ
        }
        // Digits
        for (int i = 0; i < 10; i++)
        {
            FullToHalf[(char)(0xFF10 + i)] = (char)('0' + i); // ０-９
        }
        // Space
        FullToHalf['\u3000'] = ' ';
    }

    #endregion

    #region Traditional to Simplified (common subset)

    private static readonly Dictionary<char, char> TraditionalToSimplified = BuildT2SMap();

    private static Dictionary<char, char> BuildT2SMap()
    {
        // Common traditional-simplified pairs (deduplicated).
        // Source order is preserved so the first mapping for each traditional char wins.
        const string traditional = "國學書車東風電話長門開關時間問題點頭發現場報導體認識語說話請還這個種從來過後進動員會議經濟計劃區別處機構權義務責備準確為實際關係數據擊選項設環節約條記錄製註冊號碼條項鏈構築計線網絡連結";
        const string simplified  = "国学书车东风电话长门开关时间问题点头发现场报导体认识语说话请还这个种从来过后进动员会议经济计划区别处机构权义务责备准确为实际关系数据击选项设环节约条记录制注册号码条项链构筑计线网络连结";

        var map = new Dictionary<char, char>(traditional.Length);
        int len = Math.Min(traditional.Length, simplified.Length);
        for (int i = 0; i < len; i++)
        {
            if (traditional[i] != simplified[i])
                map.TryAdd(traditional[i], simplified[i]);
        }
        return map;
    }

    #endregion

    /// <summary>
    /// Normalizes Chinese text by converting NSW (Non-Standard-Words) to speakable form.
    /// </summary>
    public static string Normalize(string text)
    {
        if (string.IsNullOrEmpty(text))
            return string.Empty;

        var sentences = SplitSentences(text);
        var sb = new StringBuilder();
        foreach (var sentence in sentences)
        {
            sb.Append(NormalizeSentence(sentence));
        }
        return sb.ToString();
    }

    /// <summary>
    /// Normalizes a single sentence.
    /// </summary>
    public static string NormalizeSentence(string sentence)
    {
        if (string.IsNullOrWhiteSpace(sentence))
            return sentence;

        sentence = ConvertTraditionalToSimplified(sentence);
        sentence = ConvertFullWidthToHalfWidth(sentence);
        sentence = NormalizeDates(sentence);
        sentence = NormalizeTimes(sentence);
        sentence = NormalizeTemperature(sentence);
        sentence = NormalizeMeasurements(sentence);
        sentence = NormalizeFractions(sentence);
        sentence = NormalizePercentages(sentence);
        sentence = NormalizePhoneNumbers(sentence);
        sentence = NormalizeRanges(sentence);
        sentence = NormalizeNumbers(sentence);
        sentence = PostReplace(sentence);
        return sentence;
    }

    private static string[] SplitSentences(string text)
    {
        var parts = ReSentenceSplit.Split(text);
        var result = new List<string>();
        for (int i = 0; i < parts.Length; i += 2)
        {
            var s = parts[i];
            if (i + 1 < parts.Length)
                s += parts[i + 1];
            if (!string.IsNullOrEmpty(s))
                result.Add(s);
        }
        return result.ToArray();
    }

    private static string ConvertTraditionalToSimplified(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            sb.Append(TraditionalToSimplified.TryGetValue(c, out var s) ? s : c);
        }
        return sb.ToString();
    }

    private static string ConvertFullWidthToHalfWidth(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (var c in text)
        {
            sb.Append(FullToHalf.TryGetValue(c, out var h) ? h : c);
        }
        return sb.ToString();
    }

    private static string NormalizeDates(string text)
    {
        text = ReDateChinese.Replace(text, m =>
        {
            var sb = new StringBuilder();
            // Year: digit by digit
            if (!string.IsNullOrEmpty(m.Groups[1].Value))
            {
                foreach (var c in m.Groups[1].Value)
                    sb.Append(NumberConverter.DigitToChinese(c));
                sb.Append('年');
            }
            // Month
            if (!string.IsNullOrEmpty(m.Groups[3].Value))
            {
                sb.Append(NumberConverter.An2Cn(m.Groups[3].Value));
                sb.Append('月');
            }
            // Day
            if (!string.IsNullOrEmpty(m.Groups[5].Value))
            {
                sb.Append(NumberConverter.An2Cn(m.Groups[5].Value));
                sb.Append(m.Value.Contains('号') ? '号' : '日');
            }
            return sb.ToString();
        });

        text = ReDateSlash.Replace(text, m =>
        {
            var sb = new StringBuilder();
            foreach (var c in m.Groups[1].Value)
                sb.Append(NumberConverter.DigitToChinese(c));
            sb.Append('年');
            sb.Append(NumberConverter.An2Cn(m.Groups[2].Value));
            sb.Append('月');
            sb.Append(NumberConverter.An2Cn(m.Groups[3].Value));
            sb.Append('日');
            return sb.ToString();
        });

        return text;
    }

    private static string NormalizeTimes(string text)
    {
        return ReTime.Replace(text, m =>
        {
            var sb = new StringBuilder();
            int hour = int.Parse(m.Groups[1].Value);
            int minute = int.Parse(m.Groups[2].Value);

            sb.Append(NumberConverter.An2Cn(hour.ToString()));
            sb.Append('点');

            if (minute == 30)
            {
                sb.Append('半');
            }
            else if (minute > 0)
            {
                if (minute < 10)
                    sb.Append('零');
                sb.Append(NumberConverter.An2Cn(minute.ToString()));
                sb.Append('分');
            }

            if (!string.IsNullOrEmpty(m.Groups[4].Value))
            {
                int second = int.Parse(m.Groups[4].Value);
                if (second > 0)
                {
                    if (second < 10)
                        sb.Append('零');
                    sb.Append(NumberConverter.An2Cn(second.ToString()));
                    sb.Append('秒');
                }
            }

            return sb.ToString();
        });
    }

    private static string NormalizeTemperature(string text)
    {
        return ReTemperature.Replace(text, m =>
        {
            var numStr = m.Groups[1].Value;
            var unit = m.Groups[2].Value;
            var sb = new StringBuilder();

            if (numStr.StartsWith('-'))
            {
                sb.Append("零下");
                numStr = numStr[1..];
            }

            if (numStr.Contains('.'))
            {
                var parts = numStr.Split('.');
                sb.Append(NumberConverter.An2Cn(parts[0]));
                sb.Append('点');
                foreach (var c in parts[1])
                    sb.Append(NumberConverter.DigitToChinese(c));
            }
            else
            {
                sb.Append(NumberConverter.An2Cn(numStr));
            }

            sb.Append(unit is "°F" ? "华氏度" : "度");
            return sb.ToString();
        });
    }

    private static string NormalizeMeasurements(string text)
    {
        // Sort by length descending to match longer units first
        foreach (var (unit, chinese) in MeasureDict.OrderByDescending(kv => kv.Key.Length))
        {
            text = text.Replace(unit, chinese);
        }
        return text;
    }

    private static string NormalizeFractions(string text)
    {
        return ReFraction.Replace(text, m =>
        {
            var numerator = m.Groups[1].Value;
            var denominator = m.Groups[2].Value;
            var sb = new StringBuilder();

            if (numerator.StartsWith('-'))
            {
                sb.Append("负");
                numerator = numerator[1..];
            }

            sb.Append(NumberConverter.An2Cn(denominator));
            sb.Append("分之");
            sb.Append(NumberConverter.An2Cn(numerator));
            return sb.ToString();
        });
    }

    private static string NormalizePercentages(string text)
    {
        return RePercentage.Replace(text, m =>
        {
            var numStr = m.Groups[1].Value;
            var sb = new StringBuilder("百分之");

            if (numStr.Contains('.'))
            {
                var parts = numStr.Split('.');
                sb.Append(NumberConverter.An2Cn(parts[0]));
                sb.Append('点');
                foreach (var c in parts[1])
                    sb.Append(NumberConverter.DigitToChinese(c));
            }
            else
            {
                sb.Append(NumberConverter.An2Cn(numStr));
            }
            return sb.ToString();
        });
    }

    private static string NormalizePhoneNumbers(string text)
    {
        text = ReMobilePhone.Replace(text, m => PhoneToString(m.Value));
        text = ReTelephone.Replace(text, m => PhoneToString(m.Value));
        return text;
    }

    private static string PhoneToString(string phone)
    {
        var sb = new StringBuilder();
        foreach (var c in phone)
        {
            if (c == '-')
                continue;
            // Use 幺 for 1 in phone numbers
            sb.Append(c == '1' ? '幺' : NumberConverter.DigitToChinese(c));
        }
        return sb.ToString();
    }

    private static string NormalizeRanges(string text)
    {
        return ReRange.Replace(text, m =>
        {
            var left = m.Groups[1].Value;
            var right = m.Groups[2].Value;
            return ConvertNumberString(left) + "到" + ConvertNumberString(right);
        });
    }

    private static string NormalizeNumbers(string text)
    {
        return ReNumber.Replace(text, m =>
        {
            return ConvertNumberString(m.Value);
        });
    }

    private static string ConvertNumberString(string numStr)
    {
        var sb = new StringBuilder();
        bool negative = numStr.StartsWith('-');
        if (negative)
        {
            sb.Append("负");
            numStr = numStr[1..];
        }

        if (numStr.Contains('.'))
        {
            var parts = numStr.Split('.');
            sb.Append(NumberConverter.An2Cn(parts[0]));
            sb.Append('点');
            foreach (var c in parts[1])
                sb.Append(NumberConverter.DigitToChinese(c));
        }
        else
        {
            sb.Append(NumberConverter.An2Cn(numStr));
        }
        return sb.ToString();
    }

    private static string PostReplace(string text)
    {
        foreach (var (greek, chinese) in GreekLetters)
        {
            text = text.Replace(greek, chinese);
        }
        return text;
    }
}
