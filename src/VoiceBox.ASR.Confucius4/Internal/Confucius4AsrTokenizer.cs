using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;

namespace VoiceBox.ASR.Confucius4.Internal
{
    /// <summary>
    /// Thin facade around <see cref="BpeTokenizer"/> for Confucius4 ASR. Owns the bundled
    /// vocab/merges and renders the static ASR chat template into a prompt string.
    /// </summary>
    /// <remarks>
    /// The Python export bakes a single chat template into <c>chat_template.json</c>
    /// that, for ASR, ends with <c>&lt;|im_start|&gt;assistant\n</c>. Rather than ship a
    /// full Jinja engine, we render the template by string assembly — the structure
    /// is fixed: one system message and one user audio turn, then the assistant
    /// generation prompt.
    ///
    /// Special tokens (<c>&lt;|audio_pad|&gt;</c>, <c>&lt;|im_start|&gt;</c>, etc.) must
    /// survive the BPE pass intact — if they get byte-level split, the count of
    /// <c>&lt;|audio_pad|&gt;</c> ids in the prompt won't match the encoder output
    /// rows and the decoder produces nothing. We mirror the tokenizer pre-
    /// tokenization regex needed by the Confucius4 tokenizer so that special tokens are
    /// recognised atomically.
    /// </remarks>
    internal sealed class Confucius4AsrTokenizer
    {
        private static readonly Regex Gpt2Regex = new(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled);

        private readonly BpeTokenizer _tokenizer;
        public string ChatTemplate { get; }

        public Confucius4AsrTokenizer(string modelDir)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelDir);
            string vocab = Path.Combine(modelDir, "vocab.json");
            string merges = Path.Combine(modelDir, "merges.txt");
            if (!File.Exists(vocab) || !File.Exists(merges))
                throw new FileNotFoundException(
                    "vocab.json / merges.txt not found in model directory.", vocab);

            var specials = LoadSpecialTokens(modelDir);
            var pre = new RegexPreTokenizer(Gpt2Regex, specials);
            var options = new BpeOptions(vocab, merges)
            {
                PreTokenizer = pre,
                ByteLevel = true,
                FuseUnknownTokens = false,
                SpecialTokens = specials,
            };
            _tokenizer = BpeTokenizer.Create(options);

            string templatePath = Path.Combine(modelDir, "chat_template.json");
            ChatTemplate = File.Exists(templatePath)
                ? ParseChatTemplate(File.ReadAllText(templatePath))
                : string.Empty;
        }

        /// <summary>
        /// Build the canonical Confucius4 prompt and tokenise it. The bundled chat
        /// template renders one audio segment as
        /// <c>&lt;|audio_start|&gt;&lt;|audio_pad|&gt;&lt;|audio_end|&gt;</c>; the Python runtime
        /// then expands the single <c>&lt;|audio_pad|&gt;</c> placeholder into
        /// <paramref name="audioFeatureRows"/> copies so the encoder hidden states
        /// align 1:1 with prompt positions. We reproduce the same expansion here.
        /// </summary>
        public int[] BuildPromptTokens(
            string audioToken,
            int audioFeatureRows,
            string context,
            string? language,
            string? prefixText)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(audioToken);

            var sb = new StringBuilder(256 + audioFeatureRows * audioToken.Length);
            sb.Append("<|im_start|>system\n").Append(context ?? string.Empty).Append("<|im_end|>\n");
            sb.Append("<|im_start|>user\n");
            sb.Append("<|audio_start|>");
            for (int i = 0; i < audioFeatureRows; i++) sb.Append(audioToken);
            sb.Append("<|audio_end|>");
            sb.Append("<|im_end|>\n");
            sb.Append("<|im_start|>assistant\n");
            if (!string.IsNullOrEmpty(language))
                sb.Append("language ").Append(language).Append("<asr_text>");
            if (!string.IsNullOrEmpty(prefixText))
                sb.Append(prefixText);

            var ids = _tokenizer.EncodeToIds(sb.ToString(),
                considerPreTokenization: true,
                considerNormalization: false);
            return ids.ToArray();
        }

        public string Decode(IEnumerable<int> ids)
            => _tokenizer.Decode(ids, considerSpecialTokens: true) ?? string.Empty;

        public int CountTokens(string text)
            => _tokenizer.CountTokens(text, considerPreTokenization: true, considerNormalization: false);

        public int[] Encode(string text)
            => _tokenizer.EncodeToIds(text, considerPreTokenization: true, considerNormalization: false).ToArray();

        private static Dictionary<string, int> LoadSpecialTokens(string modelDir)
        {
            string path = Path.Combine(modelDir, "tokenizer_config.json");
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
                    if (string.IsNullOrEmpty(token)) continue;
                    result[token] = id;
                }
            }
            return result;
        }

        private static string ParseChatTemplate(string raw)
        {
            try
            {
                using JsonDocument doc = JsonDocument.Parse(raw);
                if (doc.RootElement.TryGetProperty("chat_template", out var t))
                    return t.GetString() ?? string.Empty;
            }
            catch (JsonException) { }
            return raw;
        }
    }
}
