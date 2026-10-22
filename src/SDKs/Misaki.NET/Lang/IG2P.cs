using Misaki.NET.Models;

namespace Misaki.NET.Lang;

/// <summary>
/// Interface for grapheme-to-phoneme conversion.
/// </summary>
public interface IG2P
{
    /// <summary>
    /// Converts text to phonemes.
    /// </summary>
    /// <param name="text">Input text to convert.</param>
    /// <returns>A tuple of (phoneme string, list of tokens).</returns>
    G2PResult Process(string text);
}

/// <summary>
/// Result of G2P processing.
/// </summary>
public sealed class G2PResult
{
    /// <summary>The phoneme string output.</summary>
    public string Phonemes { get; }

    /// <summary>The list of processed tokens (may be null for some implementations).</summary>
    public IReadOnlyList<MToken>? Tokens { get; }

    public G2PResult(string phonemes, IReadOnlyList<MToken>? tokens = null)
    {
        Phonemes = phonemes;
        Tokens = tokens;
    }
}
