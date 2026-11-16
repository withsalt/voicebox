using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Misaki.NET.Jieba.Segmenter.Common;

namespace Misaki.NET.Jieba.Segmenter.PosSeg
{
    public class PosSegmenter
    {
        private static readonly WordDictionary WordDict = WordDictionary.Instance;
        private static readonly Viterbi PosSeg = Viterbi.Instance;

        // TODO: 
        private static readonly object locker = new object();

        #region Regular Expressions

        internal static readonly Regex RegexChineseInternal = new Regex(@"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%·\-]+)", RegexOptions.Compiled);
        internal static readonly Regex RegexSkipInternal = new Regex(@"(\r\n|\s)", RegexOptions.Compiled);

        internal static readonly Regex RegexChineseDetail = new Regex(@"([\u4E00-\u9FD5]+)", RegexOptions.Compiled);
        internal static readonly Regex RegexSkipDetail = new Regex(@"([\.0-9]+|[a-zA-Z0-9]+)", RegexOptions.Compiled);

        internal static readonly Regex RegexEnglishWords = new Regex(@"[a-zA-Z0-9]+", RegexOptions.Compiled);
        internal static readonly Regex RegexNumbers = new Regex(@"[\.0-9]+", RegexOptions.Compiled);

        internal static readonly Regex RegexEnglishChar = new Regex(@"^[a-zA-Z0-9]$", RegexOptions.Compiled);

        #endregion

        private static IDictionary<string, string> _wordTagTab = new Dictionary<string, string>();

        static PosSegmenter()
        {
            LoadWordTagTab();
        }

        private static void LoadWordTagTab()
        {
            try
            {
                _wordTagTab = new Dictionary<string, string>();
                using var stream = ConfigManager.OpenResource(ConfigManager.MainDictResource);
                using var reader = new StreamReader(stream, Encoding.UTF8);
                string? line;
                while ((line = reader.ReadLine()) != null)
                {
                    var tokens = line.Split(' ');
                    if (tokens.Length < 2)
                    {
                        Debug.Fail(string.Format("Invalid line: {0}", line));
                        continue;
                    }

                    var word = tokens[0];
                    var tag = tokens[2];

                    _wordTagTab[word] = tag;
                }
            }
            catch (IOException e)
            {
                Debug.Fail(string.Format("Word tag table load failure, reason: {0}", e.Message));
            }
            catch (FormatException fe)
            {
                Debug.Fail(fe.Message);
            }
        }

        private JiebaSegmenter _segmenter;

        public PosSegmenter()
        {
            _segmenter = new JiebaSegmenter();
        }

        public PosSegmenter(JiebaSegmenter segmenter)
        {
            _segmenter = segmenter;
        }

        private void CheckNewUserWordTags()
        {
            if (_segmenter.UserWordTagTab.IsNotEmpty())
            {
                _wordTagTab.Update(_segmenter.UserWordTagTab);
                _segmenter.UserWordTagTab = new Dictionary<string, string>();
            }
        }

        public IEnumerable<Pair> Cut(string text, bool hmm = true)
        {
            return CutInternal(text, hmm);
        }
        
        public IEnumerable<IEnumerable<Pair>> CutInParallel(IEnumerable<string> texts, bool hmm = true)
        {
            return texts.AsParallel().AsOrdered().Select(text => CutInternal(text, hmm));
        }
        
        public IEnumerable<IEnumerable<Pair>> CutInParallel(string text, bool hmm = true)
        {
            var lines = text.SplitLines();
            return CutInParallel(lines, hmm);
        }

        #region Internal Cut Methods

        internal IEnumerable<Pair> CutInternal(string text, bool hmm = true)
        {
            CheckNewUserWordTags();

            var blocks = RegexChineseInternal.Split(text);
            Func<string, IEnumerable<Pair>> cutMethod;
            if (hmm)
            {
                cutMethod = CutDag;
            }
            else
            {
                cutMethod = CutDagWithoutHmm;
            }

            var tokens = new List<Pair>();
            foreach (var blk in blocks)
            {
                if (RegexChineseInternal.IsMatch(blk))
                {
                    tokens.AddRange(cutMethod(blk));
                }
                else
                {
                    var tmp = RegexSkipInternal.Split(blk);
                    foreach (var x in tmp)
                    {
                        if (RegexSkipInternal.IsMatch(x))
                        {
                            tokens.Add(new Pair(x, "x"));
                        }
                        else
                        {
                            foreach (var xx in x)
                            {
                                // TODO: each char?
                                var xxs = xx.ToString();
                                if (RegexNumbers.IsMatch(xxs))
                                {
                                    tokens.Add(new Pair(xxs, "m"));
                                }
                                else if (RegexEnglishWords.IsMatch(x))
                                {
                                    tokens.Add(new Pair(xxs, "eng"));
                                }
                                else
                                {
                                    tokens.Add(new Pair(xxs, "x"));
                                }
                            }
                        }
                    }
                }
            }

            return tokens;
        }

        internal IEnumerable<Pair> CutDag(string sentence)
        {
            var dag = _segmenter.GetDag(sentence);
            var route = _segmenter.Calc(sentence, dag);

            var tokens = new List<Pair>();

            var x = 0;
            var n = sentence.Length;
            int bufStart = -1;
            int bufLen = 0;
            while (x < n)
            {
                var y = route[x].Key + 1;
                if (y - x == 1)
                {
                    if (bufLen == 0) bufStart = x;
                    bufLen++;
                }
                else
                {
                    if (bufLen > 0)
                    {
                        AddBufferToWordList(tokens, sentence.Substring(bufStart, bufLen));
                        bufLen = 0;
                    }
                    var w = sentence.Substring(x, y - x);
                    tokens.Add(new Pair(w, _wordTagTab.GetDefault(w, "x")));
                }
                x = y;
            }

            if (bufLen > 0)
            {
                AddBufferToWordList(tokens, sentence.Substring(bufStart, bufLen));
            }

            return tokens;
        }

        internal IEnumerable<Pair> CutDagWithoutHmm(string sentence)
        {
            var dag = _segmenter.GetDag(sentence);
            var route = _segmenter.Calc(sentence, dag);

            var tokens = new List<Pair>();

            var x = 0;
            int bufStart = -1;
            int bufLen = 0;
            var n = sentence.Length;

            var y = -1;
            while (x < n)
            {
                y = route[x].Key + 1;
                var w = sentence.Substring(x, y - x);
                if (RegexEnglishChar.IsMatch(w))
                {
                    if (bufLen == 0) bufStart = x;
                    bufLen += (y - x);
                    x = y;
                }
                else
                {
                    if (bufLen > 0)
                    {
                        tokens.Add(new Pair(sentence.Substring(bufStart, bufLen), "eng"));
                        bufLen = 0;
                    }
                    tokens.Add(new Pair(w, _wordTagTab.GetDefault(w, "x")));
                    x = y;
                }
            }

            if (bufLen > 0)
            {
                tokens.Add(new Pair(sentence.Substring(bufStart, bufLen), "eng"));
            }

            return tokens;
        }

        internal IEnumerable<Pair> CutDetail(string text)
        {
            var tokens = new List<Pair>();
            var blocks = RegexChineseDetail.Split(text);
            foreach (var blk in blocks)
            {
                if (RegexChineseDetail.IsMatch(blk))
                {
                    tokens.AddRange(PosSeg.Cut(blk));
                }
                else
                {
                    var tmp = RegexSkipDetail.Split(blk);
                    foreach (var x in tmp)
                    {
                        if (!string.IsNullOrWhiteSpace(x))
                        {
                            if (RegexNumbers.IsMatch(x))
                            {
                                tokens.Add(new Pair(x, "m"));
                            }
                            else if(RegexEnglishWords.IsMatch(x))
                            {
                                tokens.Add(new Pair(x, "eng"));
                            }
                            else
                            {
                                tokens.Add(new Pair(x, "x"));
                            }
                        }
                    }
                }
            }

            return tokens;
        }

        #endregion

        #region Private Helpers

        private void AddBufferToWordList(List<Pair> words, string buf)
        {
            if (buf.Length == 1)
            {
                words.Add(new Pair(buf, _wordTagTab.GetDefault(buf, "x")));
            }
            else
            {
                if (!WordDict.ContainsWord(buf))
                {
                    var tokens = CutDetail(buf);
                    words.AddRange(tokens);
                }
                else
                {
                    words.AddRange(buf.Select(ch => new Pair(ch.ToString(), "x")));
                }
            }
        }

        #endregion
    }
}