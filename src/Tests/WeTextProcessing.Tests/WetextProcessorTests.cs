using WeTextProcessing.NET;

namespace WeTextProcessing.Tests
{
    [TestClass]
    public class WetextProcessorTests
    {
        private static WetextProcessor CreateProcessor()
        {
            return new WetextProcessor();
        }

        [TestMethod]
        public void Dispose_CalledTwice_DoesNotThrow()
        {
            using var processor = CreateProcessor();
            processor.Dispose();
        }

        [TestMethod]
        public void Tag_AfterDispose_Throws()
        {
            var processor = CreateProcessor();
            processor.Dispose();
            Assert.ThrowsExactly<ObjectDisposedException>(() => processor.Tag("test"));
        }

        [TestMethod]
        public void Tag_NullInput_Throws()
        {
            using var processor = CreateProcessor();
            Assert.ThrowsExactly<ArgumentNullException>(() => processor.Tag(null!));
        }

        [TestMethod]
        public void Tag_EmptyInput_ReturnsEmpty()
        {
            using var processor = CreateProcessor();
            Assert.AreEqual(string.Empty, processor.Tag(string.Empty));
        }

        [TestMethod]
        public void Normalize_EmptyInput_ReturnsEmpty()
        {
            using var processor = CreateProcessor();
            Assert.AreEqual(string.Empty, processor.Normalize(string.Empty));
        }

        [TestMethod]
        public void Normalize_ChineseNumber_ReturnsCorrectResult()
        {
            using var processor = CreateProcessor();
            var result = processor.Normalize("123");
            Assert.AreEqual("一百二十三", result);
        }

        [TestMethod]
        public void Tag_ChineseNumber_ReturnsTaggedText()
        {
            using var processor = CreateProcessor();
            var result = processor.Tag("123");
            Assert.IsNotNull(result);
            Assert.AreNotEqual(0, result.Length);
        }

        [TestMethod]
        public void Verbalize_TaggedText_ReturnsNormalizedText()
        {
            using var processor = CreateProcessor();
            var tagged = processor.Tag("123");
            var result = processor.Verbalize(tagged);
            Assert.AreEqual("一百二十三", result);
        }

        [TestMethod]
        public void Normalize_EquivalentToTagThenVerbalize()
        {
            using var processor = CreateProcessor();
            var tagged = processor.Tag("456");
            var verbalized = processor.Verbalize(tagged);
            var normalized = processor.Normalize("456");
            Assert.AreEqual(verbalized, normalized);
        }

        [TestMethod]
        public void Normalize_DateExpression_ReturnsCorrectResult()
        {
            using var processor = CreateProcessor();
            var result = processor.Normalize("2023年5月18日");
            Assert.AreEqual("二零二三年五月十八日", result);
        }

        [TestMethod]
        public void Constructor_EnglishTextNormalization_Works()
        {
            using var processor = new WetextProcessor(WetextLanguage.English, WetextMode.TextNormalization);
            var result = processor.Normalize("$1.50");
            Assert.AreNotEqual(0, result.Length);
        }

        [TestMethod]
        public void Constructor_EnglishInverseTextNormalization_Works()
        {
            using var processor = new WetextProcessor(WetextLanguage.English, WetextMode.InverseTextNormalization);
            var result = processor.Normalize("one hundred and twenty three");
            Assert.AreEqual("123", result);
        }

        [TestMethod]
        public void Constructor_ChineseInverseTextNormalization_Works()
        {
            using var processor = new WetextProcessor(WetextLanguage.Chinese, WetextMode.InverseTextNormalization);
            var result = processor.Normalize("一百二十三");
            Assert.AreEqual("123", result);
        }

        [TestMethod]
        public void Constructor_MissingModelFile_ThrowsFileNotFound()
        {
            Assert.ThrowsExactly<FileNotFoundException>(
                () => new WetextProcessor("missing_zh_tn_tagger.fst", "missing_zh_tn_verbalizer.fst"));
        }

        [TestMethod]
        public void Constructor_InvalidFstPrefix_ThrowsWetextProcessorException()
        {
            // A real FST file whose name carries no recognized language prefix:
            // the native side must reject it instead of aborting the process.
            var source = Path.Combine(WetextProcessor.GetDefaultModelDirectory(), "zh_tn_tagger.fst");
            var bogus = Path.Combine(Path.GetTempPath(), $"badprefix_{Guid.NewGuid():N}.fst");
            File.Copy(source, bogus);
            try
            {
                Assert.ThrowsExactly<WetextProcessorException>(() => new WetextProcessor(bogus, bogus));
            }
            finally
            {
                File.Delete(bogus);
            }
        }

        [TestMethod]
        public void Constructor_CorruptFstFile_ThrowsWetextProcessorException()
        {
            var bogus = Path.Combine(Path.GetTempPath(), $"zh_tn_corrupt_{Guid.NewGuid():N}.fst");
            File.WriteAllBytes(bogus, new byte[] { 1, 2, 3, 4 });
            try
            {
                Assert.ThrowsExactly<WetextProcessorException>(() => new WetextProcessor(bogus, bogus));
            }
            finally
            {
                File.Delete(bogus);
            }
        }

        [TestMethod]
        public void Verbalize_MalformedTaggedText_ThrowsWetextProcessorException()
        {
            using var processor = CreateProcessor();
            Assert.ThrowsExactly<WetextProcessorException>(
                () => processor.Verbalize("cardinal { value: \"unterminated } "));
        }
    }
}
