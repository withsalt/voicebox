using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Misaki.NET.Jieba.Segmenter.Common;
using Newtonsoft.Json;

namespace Misaki.NET.Jieba.Segmenter.FinalSeg
{
    public class Viterbi : IFinalSeg
    {
        private static readonly Lazy<Viterbi> Lazy = new Lazy<Viterbi>(() => new Viterbi());

        // State order: B=0, M=1, E=2, S=3. Keep stable; referenced by index throughout.
        private const int NumStates = 4;
        private static readonly char[] StateChars = { 'B', 'M', 'E', 'S' };

        private static readonly Regex RegexChinese = new Regex(@"([一-鿕]+)", RegexOptions.Compiled);
        private static readonly Regex RegexSkip = new Regex(@"([a-zA-Z0-9]+(?:\.\d+)?%?)", RegexOptions.Compiled);

        // Flat arrays derived once from the JSON tables.
        private static double[] _startProbs = new double[NumStates];
        // _transProbs[from, to]
        private static double[,] _transProbs = new double[NumStates, NumStates];
        // For each target state y, the list of valid previous states y0, in the original iteration order
        // (matters for tie-breaking because the original used `<=`, i.e. later candidates win on ties).
        private static int[][] _prevStatus = new int[NumStates][];

        // Character → 4 emit probabilities (one per state), MinProb for unseen states.
        // Unknown chars use Constants.MinProb for every state.
        private static Dictionary<char, double[]> _emitProbs = new Dictionary<char, double[]>();

        private Viterbi()
        {
            LoadModel();
        }

        public static Viterbi Instance
        {
            get { return Lazy.Value; }
        }

        public IEnumerable<string> Cut(string sentence)
        {
            var tokens = new List<string>();
            foreach (var blk in RegexChinese.Split(sentence))
            {
                if (RegexChinese.IsMatch(blk))
                {
                    tokens.AddRange(ViterbiCut(blk));
                }
                else
                {
                    var segments = RegexSkip.Split(blk).Where(seg => !string.IsNullOrEmpty(seg));
                    tokens.AddRange(segments);
                }
            }
            return tokens;
        }

        #region Private Helpers

        private static int StateIndex(char c) => c switch
        {
            'B' => 0,
            'M' => 1,
            'E' => 2,
            'S' => 3,
            _ => throw new ArgumentOutOfRangeException(nameof(c), $"Unknown HMM state: {c}")
        };

        private void LoadModel()
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();

            // Preserve original iteration order for tie-breaking (last equal candidate wins).
            _prevStatus = new int[NumStates][];
            _prevStatus[0] = new[] { StateIndex('E'), StateIndex('S') }; // B ← E, S
            _prevStatus[1] = new[] { StateIndex('M'), StateIndex('B') }; // M ← M, B
            _prevStatus[2] = new[] { StateIndex('B'), StateIndex('M') }; // E ← B, M
            _prevStatus[3] = new[] { StateIndex('S'), StateIndex('E') }; // S ← S, E

            _startProbs = new double[NumStates];
            _startProbs[0] = -0.26268660809250016;   // B
            _startProbs[2] = -3.14e+100;             // E
            _startProbs[1] = -3.14e+100;             // M
            _startProbs[3] = -1.4652633398537678;    // S

            // Load trans/emit from JSON then flatten.
            var transJson = ReadResource(ConfigManager.ProbTransResource);
            var transRaw = JsonConvert.DeserializeObject<IDictionary<char, IDictionary<char, double>>>(transJson)
                ?? throw new InvalidOperationException("Failed to load trans probs.");

            var emitJson = ReadResource(ConfigManager.ProbEmitResource);
            var emitRaw = JsonConvert.DeserializeObject<IDictionary<char, IDictionary<char, double>>>(emitJson)
                ?? throw new InvalidOperationException("Failed to load emit probs.");

            _transProbs = new double[NumStates, NumStates];
            for (int from = 0; from < NumStates; from++)
            {
                for (int to = 0; to < NumStates; to++)
                    _transProbs[from, to] = Constants.MinProb;

                if (transRaw.TryGetValue(StateChars[from], out var row))
                {
                    foreach (var kv in row)
                        _transProbs[from, StateIndex(kv.Key)] = kv.Value;
                }
            }

            // Flatten emit probabilities: map char → double[4].
            // transRaw/emitRaw use state-keyed outer dicts; invert them keyed by character.
            _emitProbs = new Dictionary<char, double[]>(8192);
            for (int s = 0; s < NumStates; s++)
            {
                if (!emitRaw.TryGetValue(StateChars[s], out var charRow))
                    continue;
                foreach (var kv in charRow)
                {
                    if (!_emitProbs.TryGetValue(kv.Key, out var arr))
                    {
                        arr = new double[NumStates];
                        for (int i = 0; i < NumStates; i++) arr[i] = Constants.MinProb;
                        _emitProbs[kv.Key] = arr;
                    }
                    arr[s] = kv.Value;
                }
            }

            stopWatch.Stop();
            Debug.WriteLine("model loading finished, time elapsed {0} ms.", stopWatch.ElapsedMilliseconds);
        }

        private static string ReadResource(string resourceName)
        {
            using var stream = ConfigManager.OpenResource(resourceName);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            return reader.ReadToEnd();
        }

        private static double[] EmitFor(char c)
        {
            if (_emitProbs.TryGetValue(c, out var arr))
                return arr;
            // Unseen char: treat every state as MinProb. Allocating once per call is fine —
            // the hot path is dominated by seen chars.
            var fallback = new double[NumStates];
            for (int i = 0; i < NumStates; i++) fallback[i] = Constants.MinProb;
            return fallback;
        }

        private IEnumerable<string> ViterbiCut(string sentence)
        {
            int n = sentence.Length;
            // v[i, s] = best cumulative log-prob ending in state s at position i.
            // parent[i, s] = previous state index on that best path.
            var v = new double[n, NumStates];
            var parent = new int[n, NumStates];

            // Init
            var emit0 = EmitFor(sentence[0]);
            for (int s = 0; s < NumStates; s++)
            {
                v[0, s] = _startProbs[s] + emit0[s];
                parent[0, s] = -1;
            }

            // Recurrence
            for (int i = 1; i < n; i++)
            {
                var emit = EmitFor(sentence[i]);
                for (int y = 0; y < NumStates; y++)
                {
                    double bestScore = double.MinValue;
                    int bestPrev = 0;
                    var prevs = _prevStatus[y];
                    double emp = emit[y];
                    // Use `<=` to preserve original tie-breaking (later candidate in prevs wins).
                    for (int k = 0; k < prevs.Length; k++)
                    {
                        int y0 = prevs[k];
                        double score = v[i - 1, y0] + _transProbs[y0, y] + emp;
                        if (bestScore <= score)
                        {
                            bestScore = score;
                            bestPrev = y0;
                        }
                    }
                    v[i, y] = bestScore;
                    parent[i, y] = bestPrev;
                }
            }

            // End in E (2) or S (3), pick the higher-scoring one.
            double probE = v[n - 1, 2];
            double probS = v[n - 1, 3];
            int last = probE < probS ? 3 : 2;

            // Backtrack
            var path = new int[n];
            path[n - 1] = last;
            for (int i = n - 1; i > 0; i--)
                path[i - 1] = parent[i, path[i]];

            // Emit tokens from the state path
            var tokens = new List<string>();
            int begin = 0, next = 0;
            for (int i = 0; i < n; i++)
            {
                int pos = path[i];
                if (pos == 0)          // B
                    begin = i;
                else if (pos == 2)     // E
                {
                    tokens.Add(sentence.Sub(begin, i + 1));
                    next = i + 1;
                }
                else if (pos == 3)     // S
                {
                    tokens.Add(sentence.Sub(i, i + 1));
                    next = i + 1;
                }
            }
            if (next < n)
            {
                tokens.Add(sentence.Substring(next));
            }

            return tokens;
        }

        #endregion
    }
}
