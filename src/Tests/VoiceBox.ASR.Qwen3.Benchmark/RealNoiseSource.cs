using System.Text;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 真实噪声混入器: 从单个 wav 或目录 (递归 *.wav) 取真实录制噪声, 按目标 SNR 叠到干净 utterance,
    /// 用于量化 DNR 在<b>训练分布内</b>噪声 (DEMAND/DNS 等真实环境噪声) 上的收益。
    /// 与 <see cref="NoiseMixer"/> (白噪声, 分布外) 互补。
    ///
    /// 与白噪声路径一致的确定性保证: 每条 utterance 按其路径稳定哈希 (FNV-1a ^ seed) 播种,
    /// 选定<b>同一个</b>噪声文件、<b>同一个</b>起始偏移, 因此 none/dnr 各配置、跨进程都看到字节
    /// 相同的含噪信号, 对照公平。噪声短于语音时循环平铺; SNR 以整段功率计; 叠加后钳到 [-1,1]。
    ///
    /// 噪声数组按文件路径缓存且<b>单文件截到 <see cref="MaxCacheSeconds"/> 秒</b>, 避免 DEMAND 这类
    /// 300s 长录音把内存撑爆 (288 文件 × 300s × f32 ≈ 数 GB)。截断窗内仍随机取偏移, 多样性足够。
    /// 期望 16 kHz / mono / 16-bit PCM (复用 <see cref="WavReader"/>), 与本仓库的 DEMAND/DNS 拷贝一致。
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
                // 排序固定枚举顺序, 让 rng.Next(index) 的选择跨进程可复现。
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

            int seed = unchecked((int)StableHash(stableKey) ^ globalSeed);
            var rng = new Random(seed);

            // 1) 选文件 + 选起始偏移 (都由 seed 决定, 故同一条 utterance 恒定)。
            float[] noise = LoadNoise(_files[rng.Next(_files.Length)]);
            if (noise.Length == 0) return (float[])clean.Clone();
            int start = noise.Length > n ? rng.Next(noise.Length - n) : 0;

            // 2) 取长度 n 的噪声段, 不足则循环平铺。
            var seg = new float[n];
            for (int i = 0; i < n; i++) seg[i] = noise[(start + i) % noise.Length];

            double noisePower = 0.0;
            for (int i = 0; i < n; i++) noisePower += (double)seg[i] * seg[i];
            noisePower /= n;
            if (noisePower <= 0.0) return (float[])clean.Clone();

            // 3) 按目标 SNR 缩放并叠加, 削顶模拟采集。
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

        /// <summary>FNV-1a 32-bit, 跨进程稳定 (不像 String.GetHashCode 随机化)。与 NoiseMixer 同实现。</summary>
        private static uint StableHash(string s)
        {
            const uint offset = 2166136261;
            const uint prime = 16777619;
            uint hash = offset;
            foreach (byte b in Encoding.UTF8.GetBytes(s))
            {
                hash ^= b;
                hash *= prime;
            }
            return hash;
        }
    }
}
