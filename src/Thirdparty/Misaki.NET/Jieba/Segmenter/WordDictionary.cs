using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace Misaki.NET.Jieba.Segmenter
{
    public class WordDictionary
    {
        private static readonly Lazy<WordDictionary> lazy = new Lazy<WordDictionary>(() => new WordDictionary());

        // Internal character-trie node. `Freq > 0` marks a complete word ending at this node.
        // Children allocated lazily to keep leaf nodes small.
        internal sealed class CharTrieNode
        {
            public Dictionary<char, CharTrieNode>? Children;
            public int Freq;

            public CharTrieNode? GetChild(char c)
            {
                if (Children == null) return null;
                return Children.TryGetValue(c, out var child) ? child : null;
            }
        }

        // Root of the character trie used by GetDag / prefix walks.
        internal readonly CharTrieNode Root = new CharTrieNode();

        // Kept for backwards compatibility (SpellChecker iterates this).
        // Unlike the legacy implementation, only real words are stored (no zero-freq prefix entries),
        // which is the only case SpellChecker actually uses (`wd.Value > 0`).
        internal IDictionary<string, int> Trie = new Dictionary<string, int>();

        /// <summary>
        /// total occurrence of all words.
        /// </summary>
        public double Total { get; set; }

        private WordDictionary()
        {
            LoadDict();

            Debug.WriteLine("{0} words", Trie.Count);
            Debug.WriteLine("total freq: {0}", Total);
        }

        public static WordDictionary Instance
        {
            get { return lazy.Value; }
        }

        private void LoadDict()
        {
            try
            {
                var stopWatch = new Stopwatch();
                stopWatch.Start();

                using (var stream = ConfigManager.OpenResource(ConfigManager.MainDictResource))
                using (var sr = new StreamReader(stream, Encoding.UTF8))
                {
                    string? line;
                    while ((line = sr.ReadLine()) != null)
                    {
                        var tokens = line.Split(' ');
                        if (tokens.Length < 2)
                        {
                            Debug.Fail(string.Format("Invalid line: {0}", line));
                            continue;
                        }

                        var word = tokens[0];
                        var freq = int.Parse(tokens[1]);

                        InsertWordInternal(word, freq);
                        Total += freq;
                    }
                }

                stopWatch.Stop();
                Debug.WriteLine("main dict load finished, time elapsed {0} ms", stopWatch.ElapsedMilliseconds);
            }
            catch (IOException e)
            {
                Debug.Fail(string.Format("{0} load failure, reason: {1}", ConfigManager.MainDictResource, e.Message));
            }
            catch (FormatException fe)
            {
                Debug.Fail(fe.Message);
            }
        }

        private void InsertWordInternal(string word, int freq)
        {
            Trie[word] = freq;

            var node = Root;
            for (int i = 0; i < word.Length; i++)
            {
                node.Children ??= new Dictionary<char, CharTrieNode>();
                if (!node.Children.TryGetValue(word[i], out var child))
                {
                    child = new CharTrieNode();
                    node.Children[word[i]] = child;
                }
                node = child;
            }
            node.Freq = freq;
        }

        public bool ContainsWord(string word)
        {
            return Trie.TryGetValue(word, out var f) && f > 0;
        }

        public int GetFreqOrDefault(string key)
        {
            if (Trie.TryGetValue(key, out var f) && f > 0)
                return f;
            return 1;
        }

        /// <summary>
        /// Walks the character trie for sentence[start..start+len] and returns the word frequency,
        /// or 1 if the span is not a known word. Avoids allocating the intermediate substring.
        /// </summary>
        internal int GetFreqByRangeOrDefault(string sentence, int start, int len)
        {
            var node = Root;
            int end = start + len;
            for (int i = start; i < end; i++)
            {
                node = node.GetChild(sentence[i]);
                if (node == null) return 1;
            }
            return node.Freq > 0 ? node.Freq : 1;
        }

        public void AddWord(string word, int freq, string? tag = null)
        {
            if (Trie.TryGetValue(word, out var existing) && existing > 0)
            {
                Total -= existing;
            }

            InsertWordInternal(word, freq);
            Total += freq;
        }

        public void DeleteWord(string word)
        {
            // Keeps trie structure; zero frequency marks the node as "not a word" for lookups.
            AddWord(word, 0);
        }

        internal int SuggestFreq(string word, IEnumerable<string> segments)
        {
            double freq = 1;
            foreach (var seg in segments)
            {
                freq *= GetFreqOrDefault(seg) / Total;
            }

            return Math.Max((int)(freq * Total) + 1, GetFreqOrDefault(word));
        }
    }
}
