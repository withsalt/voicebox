using System.Text;

namespace VoiceBox.Assistant.Modules.Response.Formatting
{
    /// <summary>
    /// Accumulates streamed text into speakable sentences, splitting at punctuation once the
    /// minimum length is reached. Each push consumes its fragment immediately.
    /// </summary>
    internal sealed class SentenceSplitter
    {
        private readonly int minChars;
        private readonly StringBuilder buffer = new();

        /// <summary>Creates a splitter with a positive minimum sentence length.</summary>
        public SentenceSplitter(int minChars)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(minChars, 1);
            this.minChars = minChars;
        }

        /// <summary>
        /// Consumes a fragment and returns the completed sentences. Enumerating the result again
        /// does not alter the pending text.
        /// </summary>
        public IReadOnlyList<string> Push(string fragment)
        {
            if (string.IsNullOrEmpty(fragment))
            {
                return [];
            }

            List<string>? sentences = null;
            foreach (char character in fragment)
            {
                buffer.Append(character);
                if (IsSentenceBreak(character) && buffer.Length >= minChars)
                {
                    (sentences ??= []).Add(buffer.ToString());
                    buffer.Clear();
                }
            }
            return sentences is null ? [] : sentences;
        }

        /// <summary>
        /// Returns and clears the pending text, or null when no text remains.
        /// </summary>
        public string? Flush()
        {
            if (buffer.Length == 0)
            {
                return null;
            }
            string tail = buffer.ToString();
            buffer.Clear();
            return tail;
        }

        private static bool IsSentenceBreak(char character)
        {
            return character is '。' or '！' or '？' or '，' or '；' or '.' or '!' or '?' or ',' or ';';
        }
    }
}
