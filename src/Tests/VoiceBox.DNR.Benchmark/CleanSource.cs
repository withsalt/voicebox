namespace VoiceBox.DNR.Benchmark
{
    /// <summary>
    /// 干净语音源: 既支持 AIShell-1 风格 CSV (<c>Audio:FILE,Text:LABEL</c>, 路径相对 CSV 目录),
    /// 也支持一个含 *.wav 的目录 (递归)。统一按 seed/count 做确定性抽样, 供各后端复用同一批干净样本。
    /// CSV 采样逻辑复用自 VoiceBox.ASR.Qwen3.Benchmark.DatasetLoader。
    /// </summary>
    internal static class CleanSource
    {
        /// <summary>
        /// 加载并抽样 <paramref name="count"/> 条干净 wav 绝对路径 (count &lt;= 0 表示全部)。
        /// 同一 seed 下结果可复现 (Fisher-Yates 部分洗牌)。
        /// </summary>
        public static IReadOnlyList<string> LoadAndSample(string cleanPath, int count, int seed)
        {
            ArgumentException.ThrowIfNullOrEmpty(cleanPath);

            List<string> all;
            if (Directory.Exists(cleanPath))
            {
                all = Directory.EnumerateFiles(cleanPath, "*.wav", SearchOption.AllDirectories)
                    .OrderBy(p => p, StringComparer.Ordinal)
                    .ToList();
                if (all.Count == 0)
                    throw new FileNotFoundException($"No .wav files under clean speech directory: {cleanPath}");
            }
            else if (File.Exists(cleanPath))
            {
                all = LoadCsv(cleanPath);
                if (all.Count == 0)
                    throw new InvalidDataException($"Clean CSV contained no usable rows: {cleanPath}");
            }
            else
            {
                throw new FileNotFoundException($"Clean speech path not found (csv or dir): {cleanPath}", cleanPath);
            }

            if (count <= 0 || count >= all.Count)
                return all;

            var rng = new Random(seed);
            int total = all.Count;
            for (int i = 0; i < count; i++)
            {
                int j = rng.Next(i, total);
                (all[i], all[j]) = (all[j], all[i]);
            }
            return all.GetRange(0, count);
        }

        /// <summary>解析 AIShell 风格 CSV, 取第一列为音频相对路径, 拼成绝对路径。</summary>
        private static List<string> LoadCsv(string csvPath)
        {
            string baseDir = Path.GetDirectoryName(Path.GetFullPath(csvPath))!;
            var list = new List<string>(capacity: 8192);

            using var reader = new StreamReader(csvPath);
            string? header = reader.ReadLine();
            if (header is null)
                throw new InvalidDataException($"Empty CSV: {csvPath}");

            string? line;
            while ((line = reader.ReadLine()) is not null)
            {
                if (line.Length == 0) continue;
                int comma = line.IndexOf(',');
                string rel = (comma < 0 ? line : line.Substring(0, comma)).Trim();
                if (rel.Length == 0) continue;
                list.Add(Path.GetFullPath(Path.Combine(baseDir, rel)));
            }
            return list;
        }
    }
}
