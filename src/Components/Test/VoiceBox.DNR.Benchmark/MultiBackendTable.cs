using System.Globalization;

namespace VoiceBox.DNR.Benchmark
{
    /// <summary>末尾的跨后端横评表。列宽按最长后端名动态计算。</summary>
    internal static class MultiBackendTable
    {
        public readonly struct Row
        {
            public Row(BenchmarkSummary summary, string jsonPath, bool isError, string? error)
            {
                Summary = summary;
                JsonPath = jsonPath;
                IsError = isError;
                Error = error;
            }

            public BenchmarkSummary Summary { get; }
            public string JsonPath { get; }
            public bool IsError { get; }
            public string? Error { get; }
        }

        public static void Print(IReadOnlyList<Row> rows)
        {
            Console.WriteLine();
            Console.WriteLine("==================== DNR Backend Comparison ====================");
            if (rows.Count == 0)
            {
                Console.WriteLine("(no rows)");
                Console.WriteLine("===============================================================");
                return;
            }

            int nameW = Math.Max("Backend".Length, rows.Max(r => r.Summary.Backend.Length));
            string header =
                $"{"Backend".PadRight(nameW)}  {"SISDRi",-8}  {"segSNRi",-8}  {"CleanFid",-9}  " +
                $"{"RTF",-8}  {"p90 ms",-8}  {"algo ms",-8}  {"Wall(s)",-8}  Pairs";
            Console.WriteLine(header);
            Console.WriteLine(new string('-', header.Length));

            foreach (var r in rows)
            {
                if (r.IsError)
                {
                    Console.WriteLine($"{r.Summary.Backend.PadRight(nameW)}  ERROR: {r.Error}");
                    continue;
                }
                var s = r.Summary;
                Console.WriteLine(
                    $"{s.Backend.PadRight(nameW)}  " +
                    $"{Db(s.MeanSiSdrI),-8}  {Db(s.MeanSegSnrI),-8}  {Db(s.MeanCleanFidelity),-9}  " +
                    $"{s.AggregateRtf.ToString("F3", CultureInfo.InvariantCulture),-8}  " +
                    $"{s.LatencyP90Ms.ToString("F2", CultureInfo.InvariantCulture),-8}  " +
                    $"{s.AlgorithmicLatencyMs.ToString("F1", CultureInfo.InvariantCulture),-8}  " +
                    $"{s.WallSeconds.ToString("F1", CultureInfo.InvariantCulture),-8}  " +
                    $"{s.PairCount}");
            }

            Console.WriteLine();
            Console.WriteLine("列说明: SISDRi/segSNRi 越高越好(降噪效果); CleanFid 越高越好(干净语音保真);");
            Console.WriteLine("        RTF/p90/algo 为时间指标。dB 为单位, Pairs = utterance×SNR。");
            Console.WriteLine();
            Console.WriteLine("Per-backend JSON:");
            foreach (var r in rows)
                if (!r.IsError)
                    Console.WriteLine($"  {r.Summary.Backend}: {r.JsonPath}");
            Console.WriteLine("===============================================================");
        }

        /// <summary>格式化可空 dB 值; null → "n/a"。</summary>
        public static string Db(double? v) =>
            v is { } d && !double.IsNaN(d) && !double.IsInfinity(d)
                ? d.ToString("F2", CultureInfo.InvariantCulture)
                : "n/a";
    }
}
