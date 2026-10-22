using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Misaki.NET.Jieba.Segmenter;

namespace Misaki.NET.Jieba.Analyser
{
    public class IdfLoader
    {
        internal string IdfFilePath { get; set; }
        internal IDictionary<string, double> IdfFreq { get; set; }
        internal double MedianIdf { get; set; }

        public IdfLoader(string? idfPath = null)
        {
            IdfFilePath = string.Empty;
            IdfFreq = new Dictionary<string, double>();
            MedianIdf = 0.0;
            if (!string.IsNullOrWhiteSpace(idfPath))
            {
                SetNewPath(idfPath);
            }
        }

        /// <summary>
        /// Loads the default IDF data from the embedded resource.
        /// </summary>
        public static IdfLoader CreateDefault()
        {
            var loader = new IdfLoader();
            using var stream = ConfigManager.OpenResource(ConfigManager.IdfResource);
            loader.LoadFromReader(new StreamReader(stream, Encoding.UTF8));
            loader.IdfFilePath = "<embedded:" + ConfigManager.IdfResource + ">";
            return loader;
        }

        public void SetNewPath(string newIdfPath)
        {
            var idfPath = Path.GetFullPath(newIdfPath);
            if (IdfFilePath != idfPath)
            {
                IdfFilePath = idfPath;
                using var reader = new StreamReader(idfPath, Encoding.UTF8);
                LoadFromReader(reader);
            }
        }

        private void LoadFromReader(StreamReader reader)
        {
            IdfFreq = new Dictionary<string, double>();
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                var parts = line.Trim().Split(' ');
                var word = parts[0];
                var freq = double.Parse(parts[1]);
                IdfFreq[word] = freq;
            }

            MedianIdf = IdfFreq.Values.OrderBy(v => v).ToList()[IdfFreq.Count / 2];
        }
    }
}
