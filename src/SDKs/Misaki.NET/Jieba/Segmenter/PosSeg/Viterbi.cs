using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Misaki.NET.Jieba.Segmenter.Common;
using Newtonsoft.Json;

namespace Misaki.NET.Jieba.Segmenter.PosSeg
{
    public class Viterbi
    {
        private static readonly Lazy<Viterbi> Lazy = new Lazy<Viterbi>(() => new Viterbi());

        private static IDictionary<string, double> _startProbs = new Dictionary<string, double>();
        private static IDictionary<string, IDictionary<string, double>> _transProbs = new Dictionary<string, IDictionary<string, double>>();
        private static IDictionary<string, IDictionary<char, double>> _emitProbs = new Dictionary<string, IDictionary<char, double>>();
        private static IDictionary<char, List<string>> _stateTab = new Dictionary<char, List<string>>();

        private Viterbi()
        {
            LoadModel();
        }

        // TODO: synchronized
        public static Viterbi Instance
        {
            get { return Lazy.Value; }
        }

        public IEnumerable<Pair> Cut(string sentence)
        {
            var probPosList = ViterbiCut(sentence);
            var posList = probPosList.Item2;

            var tokens = new List<Pair>();
            int begin = 0, next = 0;
            for (var i = 0; i < sentence.Length; i++)
            {
                var parts = posList[i].Split('-');
                var charState = parts[0][0];
                var pos = parts[1];
                if (charState == 'B')
                    begin = i;
                else if (charState == 'E')
                {
                    tokens.Add(new Pair(sentence.Sub(begin, i + 1), pos));
                    next = i + 1;
                }
                else if (charState == 'S')
                {
                    tokens.Add(new Pair(sentence.Sub(i, i + 1), pos));
                    next = i + 1;
                }
            }
            if (next < sentence.Length)
            {
                tokens.Add(new Pair(sentence.Substring(next), posList[next].Split('-')[1]));
            }
            
            return tokens;
        }

        #region Private Helpers

        private static void LoadModel()
        {
            var startJson = ReadResource(ConfigManager.PosProbStartResource);
            _startProbs = JsonConvert.DeserializeObject<IDictionary<string, double>>(startJson)
                ?? throw new InvalidOperationException("Failed to load pos start probs.");

            var transJson = ReadResource(ConfigManager.PosProbTransResource);
            _transProbs = JsonConvert.DeserializeObject<IDictionary<string, IDictionary<string, double>>>(transJson)
                ?? throw new InvalidOperationException("Failed to load pos trans probs.");

            var emitJson = ReadResource(ConfigManager.PosProbEmitResource);
            _emitProbs = JsonConvert.DeserializeObject<IDictionary<string, IDictionary<char, double>>>(emitJson)
                ?? throw new InvalidOperationException("Failed to load pos emit probs.");

            var tabJson = ReadResource(ConfigManager.CharStateTabResource);
            _stateTab = JsonConvert.DeserializeObject<IDictionary<char, List<string>>>(tabJson)
                ?? throw new InvalidOperationException("Failed to load char state tab.");
        }

        private static string ReadResource(string resourceName)
        {
            using var stream = ConfigManager.OpenResource(resourceName);
            using var reader = new StreamReader(stream, System.Text.Encoding.UTF8);
            return reader.ReadToEnd();
        }

        // TODO: change sentence to obs?
        private Tuple<double, List<string>> ViterbiCut(string sentence)
        {
            var v = new List<IDictionary<string, double>>();
            var memPath = new List<IDictionary<string, string>>();

            var allStates = _transProbs.Keys.ToList();

            // Init weights and paths.
            v.Add(new Dictionary<string, Double>());
            memPath.Add(new Dictionary<string, string>());
            foreach (var state in _stateTab.GetDefault(sentence[0], allStates))
            {
                var emP = _emitProbs[state].GetDefault(sentence[0], Constants.MinProb);
                v[0][state] = _startProbs[state] + emP;
                memPath[0][state] = string.Empty;
            }

            // For each remaining char
            for (var i = 1; i < sentence.Length; ++i)
            {
                v.Add(new Dictionary<string, double>());
                memPath.Add(new Dictionary<string, string>());

                var prevStates = memPath[i - 1].Keys.Where(k => _transProbs[k].Count > 0);
                var curPossibleStates = new HashSet<string>(prevStates.SelectMany(s => _transProbs[s].Keys));

                IEnumerable<string> obsStates = _stateTab.GetDefault(sentence[i], allStates);
                obsStates = curPossibleStates.Intersect(obsStates);

                if (!obsStates.Any())
                {
                    if (curPossibleStates.Count > 0)
                    {
                        obsStates = curPossibleStates;
                    }
                    else
                    {
                        obsStates = allStates;
                    }
                }

                foreach (var y in obsStates)
                {
                    var emp = _emitProbs[y].GetDefault(sentence[i], Constants.MinProb);

                    var prob = double.MinValue;
                    var state = string.Empty;

                    foreach (var y0 in prevStates)
                    {
                        var tranp = _transProbs[y0].GetDefault(y, double.MinValue);
                        tranp = v[i - 1][y0] + tranp + emp;
                        // TODO: compare two very small values;
                        // TODO: how to deal with negative infinity
                        if (prob < tranp ||
                            (prob == tranp && string.Compare(state, y0, StringComparison.InvariantCulture) < 0))
                        {
                            prob = tranp;
                            state = y0;
                        }
                    }
                    v[i][y] = prob;
                    memPath[i][y] = state;
                }
            }

            var vLast = v.Last();
            var last = memPath.Last().Keys.Select(y => new {State = y, Prob = vLast[y]});
            var endProb = double.MinValue;
            var endState = string.Empty;
            foreach (var endPoint in last)
            {
                // TODO: compare two very small values;
                if (endProb < endPoint.Prob || 
                    (endProb == endPoint.Prob && String.Compare(endState, endPoint.State, StringComparison.InvariantCulture) < 0))
                {
                    endProb = endPoint.Prob;
                    endState = endPoint.State;
                }
            }

            var route = new string[sentence.Length];
            var n = sentence.Length - 1;
            var curState = endState;
            while(n >= 0)
            {
                route[n] = curState;
                curState = memPath[n][curState];
                n--;
            }

            return new Tuple<double, List<string>>(endProb, route.ToList());
        }

        #endregion
    }
}