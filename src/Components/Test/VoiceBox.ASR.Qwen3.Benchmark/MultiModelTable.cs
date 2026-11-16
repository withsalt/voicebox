using System.Globalization;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 批量模式末尾的横向对比表。
    /// 列宽是按本批最长 model 名动态算的, 避免 0.6B/1.7B 名字长度差异导致错位。
    /// </summary>
    internal static class MultiModelTable
    {
        public readonly struct Row
        {
            public Row(string modelName, BenchmarkSummary summary, string jsonPath, bool isError, string? error)
            {
                ModelName = modelName;
                Summary = summary;
                JsonPath = jsonPath;
                IsError = isError;
                Error = error;
            }

            public string ModelName { get; }
            public BenchmarkSummary Summary { get; }
            public string JsonPath { get; }
            public bool IsError { get; }
            public string? Error { get; }
        }

        public static void Print(IReadOnlyList<Row> rows)
        {
            Console.WriteLine();
            Console.WriteLine("==================== Multi-Model Comparison ====================");
            if (rows.Count == 0)
            {
                Console.WriteLine("(no rows)");
                Console.WriteLine("================================================================");
                return;
            }

            int nameW = Math.Max("Model".Length, rows.Max(r => r.ModelName.Length));
            string header = $"{"Model".PadRight(nameW)}  {"CER",-8}  {"Exact%",-8}  {"RTF",-8}  {"Wall(s)",-8}  Samples";
            Console.WriteLine(header);
            Console.WriteLine(new string('-', header.Length));

            foreach (var r in rows)
            {
                if (r.IsError)
                {
                    Console.WriteLine($"{r.ModelName.PadRight(nameW)}  ERROR: {r.Error}");
                    continue;
                }
                var s = r.Summary;
                string cer = CerCalculator.FormatPercent(s.AggregateCer);
                string exact = CerCalculator.FormatPercent(s.ExactMatchRate);
                string rtf = s.AggregateRtf.ToString("F3", CultureInfo.InvariantCulture);
                string wall = s.WallSeconds.ToString("F1", CultureInfo.InvariantCulture);
                string samples = (s.SampleCount - s.FailedCount).ToString(CultureInfo.InvariantCulture)
                                 + (s.FailedCount > 0 ? $" (failed {s.FailedCount})" : "");
                Console.WriteLine($"{r.ModelName.PadRight(nameW)}  {cer,-8}  {exact,-8}  {rtf,-8}  {wall,-8}  {samples}");
            }

            Console.WriteLine();
            Console.WriteLine("Per-model JSON outputs:");
            foreach (var r in rows)
                Console.WriteLine($"  {r.ModelName}: {r.JsonPath}");
            Console.WriteLine("================================================================");
        }
    }
}
