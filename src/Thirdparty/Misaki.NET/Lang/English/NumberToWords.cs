using System.Globalization;
using System.Text;

namespace Misaki.NET.Lang.English;

/// <summary>
/// Converts numbers to their English word representations.
/// Thread-safe static class replacing Python num2words for English.
/// </summary>
public static class NumberToWords
{
    private static readonly string[] Ones =
    {
        "", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"
    };

    private static readonly string[] Tens =
    {
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety"
    };

    private static readonly string[] DigitWords =
    {
        "zero", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine"
    };

    private static readonly (long Threshold, string Name)[] ScaleNumbers =
    {
        (1_000_000_000_000_000_000L, "quintillion"),
        (1_000_000_000_000_000L, "quadrillion"),
        (1_000_000_000_000L, "trillion"),
        (1_000_000_000L, "billion"),
        (1_000_000L, "million"),
        (1_000L, "thousand"),
    };

    /// <summary>
    /// Converts an integer to its English cardinal representation.
    /// </summary>
    /// <param name="number">The number to convert.</param>
    /// <returns>English words for the number (e.g., 42 → "forty-two").</returns>
    public static string ToCardinal(long number)
    {
        if (number == 0)
            return "zero";

        if (number < 0)
            return "minus " + ToCardinal(-number);

        return ConvertWholeNumber(number).Trim();
    }

    /// <summary>
    /// Converts a floating-point number to its English cardinal representation.
    /// Integer part as cardinal, then "point" followed by digit-by-digit spelling.
    /// </summary>
    /// <param name="number">The number to convert.</param>
    /// <returns>English words (e.g., 3.14 → "three point one four").</returns>
    public static string ToCardinal(double number)
    {
        if (double.IsNaN(number))
            return "not a number";
        if (double.IsPositiveInfinity(number))
            return "infinity";
        if (double.IsNegativeInfinity(number))
            return "minus infinity";

        // Check if it's effectively a whole number
        if (number == Math.Floor(number) && number >= long.MinValue && number <= long.MaxValue)
            return ToCardinal((long)number);

        bool negative = number < 0;
        if (negative)
            number = -number;

        // Split into integer and fractional parts using string to preserve digits
        string numStr = number.ToString("G", CultureInfo.InvariantCulture);
        string[] parts = numStr.Split('.');

        long integerPart = long.Parse(parts[0], CultureInfo.InvariantCulture);
        string fractionalPart = parts.Length > 1 ? parts[1] : "";

        var sb = new StringBuilder();

        if (negative)
            sb.Append("minus ");

        sb.Append(ToCardinal(integerPart));
        sb.Append(" point");

        foreach (char digit in fractionalPart)
        {
            sb.Append(' ');
            sb.Append(DigitWords[digit - '0']);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Converts an integer to its English ordinal representation.
    /// </summary>
    /// <param name="number">The number to convert.</param>
    /// <returns>English ordinal (e.g., 42 → "forty-second", 1 → "first").</returns>
    public static string ToOrdinal(long number)
    {
        if (number == 0)
            return "zeroth";

        if (number < 0)
            return "minus " + ToOrdinal(-number);

        string cardinal = ToCardinal(number);
        return ApplyOrdinalSuffix(cardinal);
    }

    /// <summary>
    /// Converts a year number to its English representation.
    /// </summary>
    /// <param name="number">The year to convert.</param>
    /// <returns>English year (e.g., 1984 → "nineteen eighty-four", 2000 → "two thousand").</returns>
    public static string ToYear(int number)
    {
        if (number < 0)
            return "minus " + ToYear(-number);

        if (number == 0)
            return "zero";

        // For 4-digit years, use special year formatting
        if (number >= 1000 && number <= 9999)
        {
            int firstHalf = number / 100;
            int secondHalf = number % 100;

            // Round thousands (2000, 3000, etc.)
            if (number % 1000 == 0)
                return ToCardinal((long)number);

            // Round centuries (1900, 1800, etc.)
            if (secondHalf == 0)
                return ConvertUnderHundred(firstHalf) + " hundred";

            string firstPart = ConvertUnderHundred(firstHalf);

            // Single digit in second half: "oh-five"
            if (secondHalf < 10)
                return firstPart + " oh-" + Ones[secondHalf];

            // Normal two-digit second half
            return firstPart + " " + ConvertUnderHundred(secondHalf);
        }

        // For non-4-digit years, just use cardinal
        return ToCardinal((long)number);
    }

    /// <summary>
    /// Converts a number less than 100 to words.
    /// </summary>
    private static string ConvertUnderHundred(int number)
    {
        if (number < 20)
            return Ones[number];

        int ten = number / 10;
        int one = number % 10;

        if (one == 0)
            return Tens[ten];

        return Tens[ten] + "-" + Ones[one];
    }

    /// <summary>
    /// Converts a number less than 1000 to words, optionally with "and".
    /// </summary>
    private static string ConvertUnderThousand(long number, bool useAnd = true)
    {
        if (number < 20)
            return Ones[number];

        if (number < 100)
            return ConvertUnderHundred((int)number);

        int hundreds = (int)(number / 100);
        long remainder = number % 100;

        string result = Ones[hundreds] + " hundred";

        if (remainder > 0)
        {
            if (useAnd)
                result += " and ";
            else
                result += " ";

            result += ConvertUnderHundred((int)remainder);
        }

        return result;
    }

    /// <summary>
    /// Converts a whole number to English words.
    /// </summary>
    private static string ConvertWholeNumber(long number)
    {
        if (number == 0)
            return "";

        if (number < 1000)
            return ConvertUnderThousand(number);

        var parts = new List<string>();
        long remaining = number;

        foreach (var (threshold, name) in ScaleNumbers)
        {
            if (remaining >= threshold)
            {
                long count = remaining / threshold;
                parts.Add(ConvertUnderThousand(count, useAnd: false) + " " + name);
                remaining %= threshold;
            }
        }

        if (remaining > 0)
        {
            // Use "and" before the last part if it's less than 100
            if (remaining < 100 && parts.Count > 0)
                parts.Add("and " + ConvertUnderThousand(remaining));
            else if (parts.Count > 0)
                parts.Add(ConvertUnderThousand(remaining));
            else
                parts.Add(ConvertUnderThousand(remaining));
        }

        return string.Join(", ", parts);
    }

    /// <summary>
    /// Applies ordinal suffix to the last word of a cardinal string.
    /// </summary>
    private static string ApplyOrdinalSuffix(string cardinal)
    {
        // Find the last word (may be hyphenated)
        // We need to transform the last "component" word
        // e.g., "forty-two" → "forty-second", "twenty" → "twentieth"

        // Irregular ordinals - check the last word
        string lastWord = GetLastWord(cardinal);

        string ordinalEnd = lastWord switch
        {
            "one" => "first",
            "two" => "second",
            "three" => "third",
            "five" => "fifth",
            "eight" => "eighth",
            "nine" => "ninth",
            "twelve" => "twelfth",
            _ => null!
        };

        if (ordinalEnd != null)
            return ReplaceLastWord(cardinal, lastWord, ordinalEnd);

        // Words ending in -y → -ieth
        if (lastWord.EndsWith("y", StringComparison.Ordinal))
        {
            string replaced = lastWord[..^1] + "ieth";
            return ReplaceLastWord(cardinal, lastWord, replaced);
        }

        // Default: append -th
        return ReplaceLastWord(cardinal, lastWord, lastWord + "th");
    }

    /// <summary>
    /// Gets the last word from a string, considering hyphens as word separators.
    /// </summary>
    private static string GetLastWord(string text)
    {
        // Split by spaces first
        int lastSpace = text.LastIndexOf(' ');
        string lastToken = lastSpace >= 0 ? text[(lastSpace + 1)..] : text;

        // Then check for hyphen
        int lastHyphen = lastToken.LastIndexOf('-');
        if (lastHyphen >= 0)
            return lastToken[(lastHyphen + 1)..];

        return lastToken;
    }

    /// <summary>
    /// Replaces the last occurrence of a word in the string.
    /// </summary>
    private static string ReplaceLastWord(string text, string oldWord, string newWord)
    {
        int lastIndex = text.LastIndexOf(oldWord, StringComparison.Ordinal);
        if (lastIndex < 0)
            return text + newWord;

        return string.Concat(text.AsSpan(0, lastIndex), newWord, text.AsSpan(lastIndex + oldWord.Length));
    }
}
