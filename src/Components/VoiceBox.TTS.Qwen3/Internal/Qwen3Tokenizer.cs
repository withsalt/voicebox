using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Qwen BBPE tokenizer for Qwen3-TTS. Renders the CustomVoice chat prompt and tokenises it.
    /// Special tokens (&lt;|im_start|&gt;, &lt;|im_end|&gt;, ...) must survive the BPE pass atomically,
    /// so we use the Qwen pre-tokenization regex + the special-token map from tokenizer_config.json
    /// (same setup as VoiceBox.ASR.Qwen3). Token-id parity vs the HF processor is covered by tests.
    /// </summary>
    internal sealed class Qwen3Tokenizer
    {
        private static readonly Regex Gpt2Regex = new(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled);

        private readonly BpeTokenizer _tokenizer;

        public Qwen3Tokenizer(string tokenizerDir)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerDir);
            string vocab = Path.Combine(tokenizerDir, "vocab.json");
            string merges = Path.Combine(tokenizerDir, "merges.txt");
            if (!File.Exists(vocab) || !File.Exists(merges))
                throw new FileNotFoundException("vocab.json / merges.txt 未找到。", vocab);

            var specials = LoadSpecialTokens(tokenizerDir);
            var options = new BpeOptions(vocab, merges)
            {
                PreTokenizer = new RegexPreTokenizer(Gpt2Regex, specials),
                ByteLevel = true,
                FuseUnknownTokens = false,
                SpecialTokens = specials,
            };
            _tokenizer = BpeTokenizer.Create(options);
        }

        /// <summary>
        /// Tokenise the assistant prompt: <c>&lt;|im_start|&gt;assistant\n{text}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>.
        /// Layout: [im_start, assistant, \n, &lt;text...&gt;, im_end, \n, im_start, assistant, \n].
        /// </summary>
        public int[] EncodeAssistant(string text)
            => Encode($"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");

        /// <summary>Tokenise the instruct prompt (1.7B only): <c>&lt;|im_start|&gt;user\n{instruct}&lt;|im_end|&gt;\n</c>.</summary>
        public int[] EncodeInstruct(string instruct)
            => Encode($"<|im_start|>user\n{instruct}<|im_end|>\n");

        private int[] Encode(string s)
            => _tokenizer.EncodeToIds(s, considerPreTokenization: true, considerNormalization: false).ToArray();

        private static Dictionary<string, int> LoadSpecialTokens(string tokenizerDir)
        {
            string path = Path.Combine(tokenizerDir, "tokenizer_config.json");
            var result = new Dictionary<string, int>(StringComparer.Ordinal);
            if (!File.Exists(path)) return result;
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.TryGetProperty("added_tokens_decoder", out var decoder))
            {
                foreach (var entry in decoder.EnumerateObject())
                {
                    if (!int.TryParse(entry.Name, out int id)) continue;
                    if (!entry.Value.TryGetProperty("content", out var content)) continue;
                    string? token = content.GetString();
                    if (!string.IsNullOrEmpty(token)) result[token] = id;
                }
            }
            return result;
        }
    }
}
