using System.Numerics.Tensors;

namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Loads the raw fp32 embedding bins and reproduces the talker input-embedding arithmetic
    /// (projected text embeddings, codec embeddings). New exports precompute
    /// text_embedding + ResizeMLP(SiLU) into a direct lookup table; older packages fall back to
    /// computing the MLP in C# for compatibility.
    ///
    /// All math stays fp32 for accuracy; the pipeline casts to fp16 only at the ONNX boundary.
    /// </summary>
    internal sealed class EmbeddingStore
    {
        private readonly float[]? _textProjected;  // [Vtxt, H]
        private readonly float[]? _textEmb;        // [Vtxt, textHidden]
        private readonly float[]? _fc1W;           // [textHidden, textHidden]
        private readonly float[]? _fc1B;           // [textHidden]
        private readonly float[]? _fc2W;           // [H, textHidden]
        private readonly float[]? _fc2B;           // [H]
        private readonly float[] _codecEmb;    // [Vc, H]
        private readonly float[] _cpEmb;       // [G-1, Vcp, H]

        private readonly int _textHidden;
        private readonly int _cpVocab;

        public int H { get; }
        public bool UsesProjectedTextEmbeddings => _textProjected is not null;

        public EmbeddingStore(string modelDir, Qwen3ModelConfig cfg)
        {
            string dir = Path.Combine(modelDir, "embeddings");
            var m = cfg.Embeddings;
            _codecEmb = LoadBin(dir, m.CodecEmbedding);
            _cpEmb = LoadBin(dir, m.CpCodecEmbedding);

            H = cfg.Talker.HiddenSize;
            _textHidden = cfg.Talker.TextHiddenSize;
            _cpVocab = cfg.CodePredictor.VocabSize;

            if (HasEntry(m.TextProjectedEmbedding) && File.Exists(Path.Combine(dir, m.TextProjectedEmbedding!.File)))
            {
                _textProjected = LoadBin(dir, m.TextProjectedEmbedding);
            }
            else
            {
                _textEmb = LoadBin(dir, m.TextEmbedding);
                _fc1W = LoadBin(dir, m.TextProjFc1Weight);
                _fc1B = LoadBin(dir, m.TextProjFc1Bias);
                _fc2W = LoadBin(dir, m.TextProjFc2Weight);
                _fc2B = LoadBin(dir, m.TextProjFc2Bias);
            }
        }

        private static bool HasEntry(EmbeddingEntry? e)
            => e is not null && !string.IsNullOrWhiteSpace(e.File) && e.Shape.Length > 0;

        private static float[] LoadBin(string dir, EmbeddingEntry e)
        {
            string path = Path.Combine(dir, e.File);
            if (!File.Exists(path))
                throw new FileNotFoundException($"嵌入文件不存在: {path}", path);
            long count = 1;
            foreach (int s in e.Shape) count *= s;
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length != count * sizeof(float))
                throw new InvalidDataException(
                    $"{e.File} 大小不符: 期望 {count * sizeof(float)} 字节, 实际 {bytes.Length}。");
            var arr = new float[count];
            Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
            return arr;
        }

        /// <summary>Copy talker codec_embedding[id] (length H) into dst.</summary>
        public void Codec(int id, Span<float> dst)
            => _codecEmb.AsSpan(id * H, H).CopyTo(dst);

        /// <summary>Copy code_predictor codec_embedding[group][id] (length H) into dst.</summary>
        public void CpGroup(int group, int id, Span<float> dst)
            => _cpEmb.AsSpan((group * _cpVocab + id) * H, H).CopyTo(dst);

        /// <summary>text token id → talker-hidden embedding (text_embedding + ResizeMLP/SiLU) into dst[H].</summary>
        public void TextProj(int id, Span<float> dst)
        {
            if (_textProjected is not null)
            {
                _textProjected.AsSpan(id * H, H).CopyTo(dst);
                return;
            }

            ReadOnlySpan<float> x = _textEmb!.AsSpan(id * _textHidden, _textHidden);
            Span<float> h1 = _textHidden <= 4096 ? stackalloc float[_textHidden] : new float[_textHidden];
            for (int o = 0; o < _textHidden; o++)
                h1[o] = TensorPrimitives.Dot(_fc1W!.AsSpan(o * _textHidden, _textHidden), x) + _fc1B![o];
            // SiLU: x * sigmoid(x)
            for (int i = 0; i < _textHidden; i++)
                h1[i] = h1[i] * (1f / (1f + MathF.Exp(-h1[i])));
            for (int o = 0; o < H; o++)
                dst[o] = TensorPrimitives.Dot(_fc2W!.AsSpan(o * _textHidden, _textHidden), h1) + _fc2B![o];
        }

        /// <summary>text token ids → [N, H] talker-hidden embeddings (row-major).</summary>
        public float[] TextProj(ReadOnlySpan<int> ids)
        {
            var outp = new float[ids.Length * H];
            for (int i = 0; i < ids.Length; i++)
                TextProj(ids[i], outp.AsSpan(i * H, H));
            return outp;
        }
    }
}
