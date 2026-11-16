using Microsoft.Extensions.Logging;
using VoiceBox.Audio.Common;
using VoiceBox.Audio.Common.Resampling;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>已就绪的克隆音色: prompt 文本 + VAE latent patch 序列 [Tp, Pz, Dz]。</summary>
    internal sealed class PreparedVoice
    {
        public required string Name { get; init; }
        public required string PromptText { get; init; }
        public required float[] PromptFeat { get; init; }   // [Tp*Pz*Dz] 行主序
        public required int PromptPatches { get; init; }
    }

    /// <summary>
    /// 音色克隆声音的注册与缓存: prompt wav → 16k 单声道 → 左填充到 patch_len 整数倍
    /// → AudioVAE encode → [Tp,Pz,Dz]。每音色只编码一次 (合成走缓存)。
    /// 调用方 (pipeline) 在合成 gate 内使用, 无需额外加锁。
    /// </summary>
    internal sealed class VoiceStore
    {
        public const string DefaultVoiceName = "default";

        private readonly Dictionary<string, VoxCpmCloneVoice> _configs;
        private readonly Dictionary<string, PreparedVoice> _prepared = new(StringComparer.OrdinalIgnoreCase);
        private readonly AudioVaeCodec _vae;
        private readonly VoxCpmModelConfig _cfg;
        private readonly ILogger? _logger;

        public VoiceStore(IEnumerable<VoxCpmCloneVoice> voices, AudioVaeCodec vae, VoxCpmModelConfig cfg, ILogger? logger)
        {
            _vae = vae;
            _cfg = cfg;
            _logger = logger;
            _configs = new Dictionary<string, VoxCpmCloneVoice>(StringComparer.OrdinalIgnoreCase);
            foreach (var v in voices)
            {
                if (string.Equals(v.Name, DefaultVoiceName, StringComparison.OrdinalIgnoreCase))
                    throw new ArgumentException($"克隆音色不能命名为保留名 \"{DefaultVoiceName}\"。");
                if (!_configs.TryAdd(v.Name, v))
                    throw new ArgumentException($"克隆音色重名: {v.Name}");
            }
        }

        public IReadOnlyCollection<string> Names => _configs.Keys;

        public bool Contains(string name) =>
            string.Equals(name, DefaultVoiceName, StringComparison.OrdinalIgnoreCase) || _configs.ContainsKey(name);

        /// <summary>取克隆音色 (default 返回 null = 无 prompt 合成)。首次使用时编码并缓存。</summary>
        public PreparedVoice? Resolve(string name)
        {
            if (string.Equals(name, DefaultVoiceName, StringComparison.OrdinalIgnoreCase))
                return null;
            if (_prepared.TryGetValue(name, out var ready))
                return ready;
            if (!_configs.TryGetValue(name, out var cfg))
                throw new ArgumentException($"未注册的音色: {name} (可用: {DefaultVoiceName}, {string.Join(", ", _configs.Keys)})");

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] wav16k = LoadPromptWav16k(cfg.PromptWavPath, _cfg.SampleRate);
            // 左填充到 patch_len 整数倍 (有效音频保持在序列末尾, 与 torch 参考一致)
            int patchLen = _cfg.SamplesPerStep;
            int rem = wav16k.Length % patchLen;
            if (rem != 0)
            {
                var padded = new float[wav16k.Length + (patchLen - rem)];
                Array.Copy(wav16k, 0, padded, patchLen - rem, wav16k.Length);
                wav16k = padded;
            }
            float[] latents = _vae.Encode(wav16k);              // [Dz, T] 行主序
            int frames = wav16k.Length / _cfg.VaeChunkSize;     // T
            int patches = frames / _cfg.PatchSize;              // Tp
            var prepared = new PreparedVoice
            {
                Name = cfg.Name,
                PromptText = cfg.PromptText,
                PromptFeat = TransposeToPatches(latents, frames),
                PromptPatches = patches,
            };
            _prepared[name] = prepared;
            _logger?.LogInformation("音色 {Voice} prompt 编码完成: {Sec:F2}s 音频 → {Patches} patches, 用时 {Ms} ms",
                name, (double)wav16k.Length / _cfg.SampleRate, patches, sw.ElapsedMilliseconds);
            return prepared;
        }

        /// <summary>[Dz,T] → [Tp,Pz,Dz] (对应 torch 的 view(Dz,-1,Pz).permute(1,2,0))。</summary>
        private float[] TransposeToPatches(float[] latents, int frames)
        {
            int dz = _cfg.FeatDim, pz = _cfg.PatchSize;
            int tp = frames / pz;
            var outp = new float[tp * pz * dz];
            for (int d = 0; d < dz; d++)
                for (int t = 0; t < frames; t++)
                    outp[(t / pz) * pz * dz + (t % pz) * dz + d] = latents[d * frames + t];
            return outp;
        }

        /// <summary>读 wav → 单声道 (均值) → 重采样到 targetRate (Speex q10)。</summary>
        internal static float[] LoadPromptWav16k(string path, int targetRate)
        {
            var (samples, sampleRate) = WavReader.ReadMono(path);
            if (samples.Length == 0)
                throw new InvalidDataException($"prompt WAV 不包含音频样本: {path}");
            if (sampleRate == targetRate) return samples;

            using var rs = new StreamingSpeexResampler(sampleRate, targetRate, quality: 10);
            var output = new float[rs.EstimateOutputLength(samples.Length) + 256];
            int n = rs.Process(samples, ref output);
            var tail = new float[256];
            int m = rs.Flush(ref tail);
            var all = new float[n + m];
            Array.Copy(output, all, n);
            Array.Copy(tail, 0, all, n, m);
            if (all.Length == 0)
                throw new InvalidDataException($"prompt WAV 重采样后不包含音频样本: {path}");
            return all;
        }
    }
}
