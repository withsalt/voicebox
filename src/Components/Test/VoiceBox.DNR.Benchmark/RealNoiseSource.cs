namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// 真实噪声混入器: 从单个 wav 或目录 (递归 *.wav) 取真实录制噪声, 按目标 SNR 叠到干净 utterance,
    /// 量化 DNR 在<b>训练分布内</b>噪声 (DEMAND/DNS 等真实环境噪声) 上的降噪效果。与 <see cref="NoiseMixer"/>
    /// (白噪声, 分布外) 互补 —— 神经 SE 模型对白噪声属分布外, 真实噪声下的数才有代表性。
    ///
    /// 确定性: 每条 utterance 按 <paramref name="stableKey"/> (含路径+SNR) 的 FNV-1a 哈希 ^ seed 播种,
    /// 选定<b>同一个</b>噪声文件、<b>同一个</b>起始偏移, 因此各后端、跨进程都看到字节相同的含噪信号,
    /// 横评公平。噪声短于语音时循环平铺; SNR 以整段功率计; 叠加后钳到 [-1,1]。
    /// 单文件缓存并截到 <see cref="MaxCacheSeconds"/> 秒, 避免 DEMAND 长录音撑爆内存。
    /// 期望 16 kHz / mono / 16-bit PCM (复用 <see cref="WavReader"/>)。
    /// 复制自 VoiceBox.ASR.Qwen3.Benchmark.RealNoiseSource。
    /// </summary>
    internal sealed class RealNoiseSource
    {
        private const int MaxCacheSeconds = 30;
        private const int MaxCacheSamples = WavReader.ExpectedSampleRate * MaxCacheSeconds;

        private readonly string[] _files;
        private readonly Dictionary<string, float[]> _cache = new(StringComparer.OrdinalIgnoreCase);

        public int FileCount => _files.Length;
        public string Root { get; }

        public RealNoiseSource(string pathOrDir)
        {
            ArgumentException.ThrowIfNullOrEmpty(pathOrDir);
            Root = pathOrDir;

            if (Directory.Exists(pathOrDir))
            {
                _files = Directory.EnumerateFiles(pathOrDir, "*.wav", SearchOption.AllDirectories)
                    .OrderBy(p => p, StringComparer.Ordinal)
                    .ToArray();
                if (_files.Length == 0)
                    throw new FileNotFoundException($"No .wav files under noise directory: {pathOrDir}");
            }
            else if (File.Exists(pathOrDir))
            {
                _files = new[] { pathOrDir };
            }
            else
            {
                throw new FileNotFoundException($"Noise path not found: {pathOrDir}", pathOrDir);
            }
        }

        /// <summary>返回 clean + 缩放真实噪声 (新数组)。<paramref name="snrDb"/> 越小噪声越强。</summary>
        public float[] AddNoise(float[] clean, double snrDb, string stableKey, int globalSeed)
        {
            ArgumentNullException.ThrowIfNull(clean);
            if (clean.Length == 0) return clean;

            int n = clean.Length;
            double signalPower = 0.0;
            for (int i = 0; i < n; i++) signalPower += (double)clean[i] * clean[i];
            signalPower /= n;
            if (signalPower <= 0.0) return (float[])clean.Clone(); // 全静音, 无可定义 SNR

            int seed = unchecked((int)NoiseMixer.StableHash(stableKey) ^ globalSeed);
            var rng = new Random(seed);

            float[] noise = LoadNoise(_files[rng.Next(_files.Length)]);
            if (noise.Length == 0) return (float[])clean.Clone();
            int start = noise.Length > n ? rng.Next(noise.Length - n) : 0;

            var seg = new float[n];
            for (int i = 0; i < n; i++) seg[i] = noise[(start + i) % noise.Length];

            double noisePower = 0.0;
            for (int i = 0; i < n; i++) noisePower += (double)seg[i] * seg[i];
            noisePower /= n;
            if (noisePower <= 0.0) return (float[])clean.Clone();

            double targetNoisePower = signalPower / Math.Pow(10.0, snrDb / 10.0);
            float scale = (float)Math.Sqrt(targetNoisePower / noisePower);

            var outp = new float[n];
            for (int i = 0; i < n; i++)
            {
                float v = clean[i] + scale * seg[i];
                outp[i] = v > 1f ? 1f : (v < -1f ? -1f : v);
            }
            return outp;
        }

        private float[] LoadNoise(string path)
        {
            if (_cache.TryGetValue(path, out var cached)) return cached;

            float[] samples = WavReader.ReadMonoFloat32(path, out _, out _);
            if (samples.Length > MaxCacheSamples)
            {
                var trimmed = new float[MaxCacheSamples];
                Array.Copy(samples, trimmed, MaxCacheSamples);
                samples = trimmed;
            }
            _cache[path] = samples;
            return samples;
        }
    }
}
