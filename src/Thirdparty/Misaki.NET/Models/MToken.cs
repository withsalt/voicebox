namespace Misaki.NET.Models;

/// <summary>
/// Represents a processed token with text, phonemes, and metadata.
/// </summary>
public sealed class MToken
{
    /// <summary>The original text of this token.</summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>Part-of-speech tag.</summary>
    public string Tag { get; set; } = string.Empty;

    /// <summary>Trailing whitespace after this token.</summary>
    public string Whitespace { get; set; } = string.Empty;

    /// <summary>Phoneme transcription (null if not yet resolved).</summary>
    public string? Phonemes { get; set; }

    /// <summary>Optional start timestamp.</summary>
    public double? StartTs { get; set; }

    /// <summary>Optional end timestamp.</summary>
    public double? EndTs { get; set; }

    /// <summary>Extended properties for internal processing.</summary>
    public TokenProperties Properties { get; set; } = new();
}
