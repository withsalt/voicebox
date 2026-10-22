namespace VoiceBox.ASR.Qwen3.Benchmark
{
    internal sealed record DatasetEntry(string AudioPath, string Label);

    /// <summary>
    /// 加载 AIShell-1 测试集 CSV (aishell1_test.csv)。CSV 头形如:
    ///   Audio:FILE,Text:LABEL
    ///   speech_asr_aishell1_testsets/wav/test/.../xxx.wav,识别文本
    /// </summary>
    internal static class DatasetLoader
    {
        /// <summary>
        /// 加载 CSV 并按 seed 抽样 count 条 (count &lt;= 0 表示全部, count &gt; total 则取 total)。
        /// 采样使用 Fisher-Yates, 保证同一 seed 下结果可复现; 路径相对 CSV 所在目录解析。
        /// </summary>
        public static IReadOnlyList<DatasetEntry> LoadAndSample(string csvPath, int count, int seed)
        {
            ArgumentException.ThrowIfNullOrEmpty(csvPath);
            if (!File.Exists(csvPath))
                throw new FileNotFoundException("AIShell test CSV not found.", csvPath);

            string baseDir = Path.GetDirectoryName(Path.GetFullPath(csvPath))!;
            var all = new List<DatasetEntry>(capacity: 8192);

            using (var reader = new StreamReader(csvPath))
            {
                string? header = reader.ReadLine();
                if (header is null)
                    throw new InvalidDataException($"Empty CSV: {csvPath}");

                string? line;
                while ((line = reader.ReadLine()) is not null)
                {
                    if (line.Length == 0) continue;
                    int comma = line.IndexOf(',');
                    if (comma < 0) continue;
                    string rel = line.Substring(0, comma).Trim();
                    string label = line.Substring(comma + 1).Trim();
                    if (rel.Length == 0 || label.Length == 0) continue;
                    string abs = Path.GetFullPath(Path.Combine(baseDir, rel));
                    all.Add(new DatasetEntry(abs, label));
                }
            }

            if (all.Count == 0)
                throw new InvalidDataException($"CSV contained no usable rows: {csvPath}");

            if (count <= 0 || count >= all.Count)
                return all;

            // Reservoir-style shuffle of the first `count` slots is enough; we just want a deterministic subset.
            var rng = new Random(seed);
            int total = all.Count;
            for (int i = 0; i < count; i++)
            {
                int j = rng.Next(i, total);
                (all[i], all[j]) = (all[j], all[i]);
            }
            return all.GetRange(0, count);
        }
    }
}
