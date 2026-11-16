namespace Misaki.NET.Jieba.Segmenter
{
    using System.IO;

    /// <summary>
    /// Provides access to jieba.NET model/data files as embedded resource streams.
    /// Resources live under Misaki.NET.Resources.Jieba.* in the Misaki assembly.
    /// </summary>
    public static class ConfigManager
    {
        private const string ResourcePrefix = "Misaki.NET.Resources.Jieba.";

        internal const string MainDictResource = "dict.txt";
        internal const string ProbTransResource = "prob_trans.json";
        internal const string ProbEmitResource = "prob_emit.json";
        internal const string PosProbStartResource = "pos_prob_start.json";
        internal const string PosProbTransResource = "pos_prob_trans.json";
        internal const string PosProbEmitResource = "pos_prob_emit.json";
        internal const string CharStateTabResource = "char_state_tab.json";
        internal const string IdfResource = "idf.txt";
        internal const string StopWordsResource = "stopwords.txt";

        /// <summary>
        /// Opens an embedded jieba resource by its file name (e.g. "dict.txt").
        /// </summary>
        public static Stream OpenResource(string resourceName)
        {
            var assembly = typeof(ConfigManager).Assembly;
            var fullName = ResourcePrefix + resourceName;
            return assembly.GetManifestResourceStream(fullName)
                ?? throw new FileNotFoundException(
                    $"Embedded jieba resource '{fullName}' not found. Available: " +
                    string.Join(", ", assembly.GetManifestResourceNames()));
        }
    }
}
