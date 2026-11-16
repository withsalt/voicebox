using System.Globalization;
using System.Text.Json;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    /// <summary>
    /// 把当前一次 benchmark 结果与一个 baseline JSON 做对照。
    /// 用 audio_path 做匹配, 跨 --seed / --count 也能取交集。
    /// </summary>
    internal static class BenchmarkComparer
    {
        public static ComparisonResult Compare(BenchmarkSummary current, string baselinePath, int topN = 5)
        {
            if (!File.Exists(baselinePath))
                throw new FileNotFoundException("Baseline JSON not found.", baselinePath);

            string json = File.ReadAllText(baselinePath);
            var baseline = JsonSerializer.Deserialize<BenchmarkSummary>(json)
                ?? throw new InvalidDataException($"Baseline JSON could not be parsed: {baselinePath}");

            return Compare(current, baseline, Path.GetFullPath(baselinePath), topN);
        }

        /// <summary>
        /// 与一个已在内存里的 baseline summary 直接对照 (矩阵模式用: none 路结果即 baseline,
        /// 无需落 JSON 再读回)。<paramref name="baselineLabel"/> 仅用于在输出里标识 baseline 来源。
        /// </summary>
        public static ComparisonResult Compare(BenchmarkSummary current, BenchmarkSummary baseline, string baselineLabel, int topN = 5)
        {            var baselineByPath = new Dictionary<string, SampleResult>(StringComparer.OrdinalIgnoreCase);
            foreach (var s in baseline.Samples)
            {
                if (s.Error is not null) continue;
                baselineByPath[s.AudioPath] = s;
            }

            int matched = 0;
            var candidates = new List<RegressedSample>();
            foreach (var cur in current.Samples)
            {
                if (cur.Error is not null) continue;
                if (!baselineByPath.TryGetValue(cur.AudioPath, out var bse)) continue;
                matched++;

                bool exactFlip = bse.ExactMatch && !cur.ExactMatch;
                double deltaCer = cur.Cer - bse.Cer;
                // 只关心退步: exact 翻车 或 CER 上升。
                if (!exactFlip && deltaCer <= 0) continue;

                candidates.Add(new RegressedSample
                {
                    AudioPath = cur.AudioPath,
                    Reference = cur.Reference,
                    BaselineHypothesis = bse.Hypothesis,
                    CurrentHypothesis = cur.Hypothesis,
                    BaselineCer = bse.Cer,
                    CurrentCer = cur.Cer,
                    DeltaCer = deltaCer,
                    BaselineExactMatch = bse.ExactMatch,
                    CurrentExactMatch = cur.ExactMatch,
                });
            }

            // exact-flip 优先 (true 排在前), 然后按 DeltaCer 降序; 平手时按 AudioPath 稳定排序。
            candidates.Sort((a, b) =>
            {
                int flipA = a.BaselineExactMatch && !a.CurrentExactMatch ? 0 : 1;
                int flipB = b.BaselineExactMatch && !b.CurrentExactMatch ? 0 : 1;
                if (flipA != flipB) return flipA - flipB;
                int byDelta = b.DeltaCer.CompareTo(a.DeltaCer);
                if (byDelta != 0) return byDelta;
                return string.CompareOrdinal(a.AudioPath, b.AudioPath);
            });

            var top = candidates.Take(topN).ToList();

            return new ComparisonResult
            {
                BaselinePath = baselineLabel,
                BaselineTimestamp = baseline.Timestamp,
                BaselineSampleCount = baseline.SampleCount,
                MatchedSampleCount = matched,

                BaselineAggregateCer = baseline.AggregateCer,
                CurrentAggregateCer = current.AggregateCer,
                DeltaAggregateCer = current.AggregateCer - baseline.AggregateCer,

                BaselineAggregateRtf = baseline.AggregateRtf,
                CurrentAggregateRtf = current.AggregateRtf,
                DeltaAggregateRtf = current.AggregateRtf - baseline.AggregateRtf,

                BaselineExactMatchRate = baseline.ExactMatchRate,
                CurrentExactMatchRate = current.ExactMatchRate,
                DeltaExactMatchRate = current.ExactMatchRate - baseline.ExactMatchRate,

                RegressedSamples = top,
            };
        }

        public static void PrintToConsole(ComparisonResult c)
        {
            Console.WriteLine();
            Console.WriteLine("==================== Comparison vs Baseline ====================");
            Console.WriteLine($"Baseline           : {c.BaselinePath}");
            Console.WriteLine($"Baseline timestamp : {c.BaselineTimestamp}");
            Console.WriteLine($"Matched samples    : {c.MatchedSampleCount} / baseline {c.BaselineSampleCount}");
            Console.WriteLine();
            Console.WriteLine($"Aggregate CER      : {Pct(c.BaselineAggregateCer)}  →  {Pct(c.CurrentAggregateCer)}   Δ {SignedPct(c.DeltaAggregateCer)} {Verdict(c.DeltaAggregateCer, lowerIsBetter: true)}");
            Console.WriteLine($"Aggregate RTF      : {c.BaselineAggregateRtf.ToString("F3", CultureInfo.InvariantCulture)}  →  {c.CurrentAggregateRtf.ToString("F3", CultureInfo.InvariantCulture)}   Δ {SignedNum(c.DeltaAggregateRtf, "F3")} {Verdict(c.DeltaAggregateRtf, lowerIsBetter: true)}");
            Console.WriteLine($"Exact-match rate   : {Pct(c.BaselineExactMatchRate)}  →  {Pct(c.CurrentExactMatchRate)}   Δ {SignedPct(c.DeltaExactMatchRate)} {Verdict(c.DeltaExactMatchRate, lowerIsBetter: false)}");

            if (c.RegressedSamples.Count == 0)
            {
                Console.WriteLine();
                Console.WriteLine("No regressed samples (no exact→error flips and no CER increases).");
            }
            else
            {
                Console.WriteLine();
                Console.WriteLine($"Top {c.RegressedSamples.Count} regressed samples:");
                int i = 1;
                foreach (var r in c.RegressedSamples)
                {
                    string flag = r.BaselineExactMatch && !r.CurrentExactMatch ? "[EXACT→ERR]" : "[CER↑]";
                    Console.WriteLine($"  {i}. {flag} ΔCER {SignedPct(r.DeltaCer)}  ({Pct(r.BaselineCer)} → {Pct(r.CurrentCer)})");
                    Console.WriteLine($"     ref : {r.Reference}");
                    Console.WriteLine($"     base: {r.BaselineHypothesis}");
                    Console.WriteLine($"     curr: {r.CurrentHypothesis}");
                    Console.WriteLine($"     wav : {r.AudioPath}");
                    i++;
                }
            }
            Console.WriteLine("================================================================");
        }

        private static string Pct(double v) => CerCalculator.FormatPercent(v);

        private static string SignedPct(double v)
        {
            string sign = v > 0 ? "+" : (v < 0 ? "" : "±");
            return sign + CerCalculator.FormatPercent(v);
        }

        private static string SignedNum(double v, string fmt)
        {
            string sign = v > 0 ? "+" : (v < 0 ? "" : "±");
            return sign + v.ToString(fmt, CultureInfo.InvariantCulture);
        }

        private static string Verdict(double delta, bool lowerIsBetter)
        {
            if (delta == 0) return "(unchanged)";
            bool better = lowerIsBetter ? delta < 0 : delta > 0;
            return better ? "(better)" : "(WORSE)";
        }
    }
}
