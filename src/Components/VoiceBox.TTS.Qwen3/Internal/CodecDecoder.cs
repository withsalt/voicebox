using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.Qwen3.Internal
{
    /// <summary>
    /// Codec decoder (audio codes → 24kHz waveform). Always fp32. Replicates
    /// Qwen3TTSTokenizerV2Decoder.chunked_decode(chunk_size, left_context) exactly.
    /// </summary>
    internal sealed class CodecDecoder : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _numGroups;
        private readonly int _samplesPerFrame;
        private readonly RunOptions _run = new();
        private static readonly string[] OutNames = ["wav"];

        public CodecDecoder(string modelPath, SessionOptions so, Qwen3ModelConfig cfg)
        {
            _session = new InferenceSession(modelPath, so);
            _numGroups = cfg.NumCodeGroups;
            _samplesPerFrame = cfg.SamplesPerFrame;
        }

        /// <summary>Decode one contiguous chunk of codes [numGroups, len] → wav[len*samplesPerFrame].</summary>
        public float[] DecodeChunk(long[] codesGT, int len)
        {
            using var input = OrtValue.CreateTensorValueFromMemory(codesGT, [1, _numGroups, len]);
            using var outputs = _session.Run(_run, ["codes"], [input], OutNames);
            return outputs[0].GetTensorDataAsSpan<float>().ToArray();
        }

        /// <summary>Decode a contiguous run of frames as a single chunk → full wav (no windowing).</summary>
        public float[] DecodeFramesContiguous(IReadOnlyList<int[]> frames)
        {
            int T = frames.Count;
            if (T == 0) return [];
            return DecodeChunk(BuildTransposed(frames, T), T);
        }

        /// <summary>
        /// Full chunked decode of all frames. <paramref name="frames"/>[t] holds the 16 codes of frame t.
        /// </summary>
        public float[] Decode(IReadOnlyList<int[]> frames, int chunkSize = 300, int leftContext = 25)
        {
            int T = frames.Count;
            if (T == 0) return [];
            long[] codesGT = BuildTransposed(frames, T);

            var output = new List<float[]>();
            int total = 0;
            int start = 0;
            while (start < T)
            {
                int end = Math.Min(start + chunkSize, T);
                int ctx = start - leftContext > 0 ? leftContext : start;
                int from = start - ctx, len = end - from;
                long[] chunk = SliceColumns(codesGT, T, from, len);
                float[] wav = DecodeChunk(chunk, len);
                int drop = ctx * _samplesPerFrame;
                output.Add(wav.AsSpan(drop).ToArray());
                total += wav.Length - drop;
                start = end;
            }
            return Concat(output, total);
        }

        private long[] BuildTransposed(IReadOnlyList<int[]> frames, int T)
        {
            var codesGT = new long[_numGroups * T];   // [G, T]
            for (int t = 0; t < T; t++)
            {
                int[] f = frames[t];
                for (int gp = 0; gp < _numGroups; gp++)
                    codesGT[gp * T + t] = Math.Max(0, f[gp]);
            }
            return codesGT;
        }

        private long[] SliceColumns(long[] codesGT, int T, int from, int len)
        {
            var chunk = new long[_numGroups * len];
            for (int gp = 0; gp < _numGroups; gp++)
                Array.Copy(codesGT, gp * T + from, chunk, gp * len, len);
            return chunk;
        }

        private static float[] Concat(List<float[]> parts, int total)
        {
            var outp = new float[total];
            int o = 0;
            foreach (var p in parts) { p.CopyTo(outp, o); o += p.Length; }
            return outp;
        }

        public void Dispose()
        {
            _run.Dispose();
            _session.Dispose();
        }
    }
}
