namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Reproduces Qwen3TTSForConditionalGeneration.generate() prompt construction for CustomVoice
    /// (non_streaming_mode=True). Byte-for-byte port of the validated Python _engine.build_prompt
    /// (prompt-construction parity 141 dB). Returns (inputsEmbeds[Tp*H] row-major, ttsPad[H]).
    /// </summary>
    internal static class PromptBuilder
    {
        public static (float[] prompt, float[] pad) Build(
            EmbeddingStore store, Qwen3ModelConfig cfg, int[] textIds,
            string speaker, string language, int[]? instructIds)
        {
            int H = store.H;
            var sp = cfg.SpecialTokens;

            float[] bos = TextRow(store, sp.TtsBos, H);
            float[] eos = TextRow(store, sp.TtsEos, H);
            float[] pad = TextRow(store, sp.TtsPad, H);
            float[] codecPad = CodecRow(store, sp.CodecPad, H);

            // language id (+ dialect override for dialect speakers in chinese/auto)
            string lang = (language ?? "auto").ToLowerInvariant();
            int? langId = lang == "auto" ? null : cfg.CodecLanguageId[lang];
            if (!string.IsNullOrEmpty(speaker) && (lang == "chinese" || lang == "auto")
                && cfg.SpeakerIsDialect.TryGetValue(speaker, out string? dialect) && dialect is not null)
                langId = cfg.CodecLanguageId[dialect];

            // codec control + speaker + (pad, bos)
            var codecIn = new List<float[]>();
            if (langId is null)
            {
                codecIn.Add(CodecRow(store, sp.CodecNothink, H));
                codecIn.Add(CodecRow(store, sp.CodecThinkBos, H));
                codecIn.Add(CodecRow(store, sp.CodecThinkEos, H));
            }
            else
            {
                codecIn.Add(CodecRow(store, sp.CodecThink, H));
                codecIn.Add(CodecRow(store, sp.CodecThinkBos, H));
                codecIn.Add(CodecRow(store, langId.Value, H));
                codecIn.Add(CodecRow(store, sp.CodecThinkEos, H));
            }
            if (!string.IsNullOrEmpty(speaker))
                codecIn.Add(CodecRow(store, cfg.SpeakerId[speaker], H));
            codecIn.Add(CodecRow(store, sp.CodecPad, H));
            codecIn.Add(CodecRow(store, sp.CodecBos, H));
            int codecLen = codecIn.Count;

            var rows = new List<float[]>();

            // optional 1.7B instruct prefix
            if (instructIds is { Length: > 0 })
                foreach (int id in instructIds) rows.Add(TextRow(store, id, H));

            // role: <|im_start|>assistant\n  (textIds[:3])
            for (int i = 0; i < 3; i++) rows.Add(TextRow(store, textIds[i], H));

            // A = (pad×(codecLen-2) ++ bos) + codecIn[:-1]
            for (int i = 0; i < codecLen - 1; i++)
            {
                float[] textSide = i < codecLen - 2 ? pad : bos;
                rows.Add(Add(textSide, codecIn[i]));
            }

            // C = (textProj(textBody) ++ eos) + codecPad   (textBody = textIds[3:-5])
            int bodyLen = textIds.Length - 8;   // drop first 3 and last 5
            for (int j = 0; j < bodyLen; j++)
                rows.Add(Add(TextRow(store, textIds[3 + j], H), codecPad));
            rows.Add(Add(eos, codecPad));

            // D = pad + codecBos
            rows.Add(Add(pad, CodecRow(store, sp.CodecBos, H)));

            var prompt = new float[rows.Count * H];
            for (int r = 0; r < rows.Count; r++) rows[r].CopyTo(prompt, r * H);
            return (prompt, pad);
        }

        private static float[] TextRow(EmbeddingStore store, int id, int H)
        {
            var r = new float[H];
            store.TextProj(id, r);
            return r;
        }

        private static float[] CodecRow(EmbeddingStore store, int id, int H)
        {
            var r = new float[H];
            store.Codec(id, r);
            return r;
        }

        private static float[] Add(float[] a, float[] b)
        {
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = a[i] + b[i];
            return r;
        }
    }
}
