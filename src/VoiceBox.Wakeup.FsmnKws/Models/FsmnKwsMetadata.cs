using System.Text.Json;

namespace VoiceBox.Wakeup.FsmnKws.Models
{
    /// <summary>
    /// 模型目录中 kws_meta.json 的解析结果 (由 scripts/kws-xiaoyun-onnx-export 随模型一并导出)。
    /// <para>
    /// 唤醒词及其 CTC token id 序列在导出时由模型确定并写入该文件, 运行期直接读取,
    /// 不再由调用方手动配置。CMVN 从 <c>cmvn.add_shift</c> / <c>cmvn.rescale</c> 读取。
    /// 文件缺失或缺少必要字段 (<c>keyword</c> / <c>keyword_token_ids</c> / <c>cmvn</c>)
    /// 时 <see cref="Load"/> 抛异常。
    /// </para>
    /// </summary>
    public sealed class FsmnKwsMetadata
    {
        private const int CmvnDimension = 400;

        /// <summary>元数据文件名 (位于模型目录, 与 kws_streaming.onnx 同级)。</summary>
        public const string FileName = "kws_meta.json";

        /// <summary>唤醒词原文 (命中时通过事件原样上报)。</summary>
        public string Keyword { get; }

        /// <summary>唤醒词逐字映射的 CTC token id 序列 (导出期已编码, 与模型 token 表对应)。</summary>
        public IReadOnlyList<int> KeywordTokenIds { get; }

        /// <summary>
        /// 无法编码的 OOV 字 (不在模型 token 表中, 这些字无法被唤醒); 正常应为空。
        /// </summary>
        public IReadOnlyList<string> OovChars { get; }

        /// <summary>CMVN 负均值，固定 400 维。</summary>
        public IReadOnlyList<float> CmvnAddShift { get; }

        /// <summary>CMVN 逆标准差，固定 400 维。</summary>
        public IReadOnlyList<float> CmvnRescale { get; }

        private FsmnKwsMetadata(string keyword, int[] tokenIds, string[] oovChars,
            float[] addShift, float[] rescale)
        {
            Keyword = keyword;
            KeywordTokenIds = tokenIds;
            OovChars = oovChars;
            CmvnAddShift = Array.AsReadOnly(addShift);
            CmvnRescale = Array.AsReadOnly(rescale);
        }

        /// <summary>
        /// 给定模型路径 (kws_streaming.onnx) 或模型目录, 返回同级 <see cref="FileName"/> 的完整路径。
        /// </summary>
        public static string ResolveMetaPath(string modelPathOrDir)
        {
            if (string.IsNullOrWhiteSpace(modelPathOrDir))
                throw new ArgumentException("modelPathOrDir must not be empty.", nameof(modelPathOrDir));
            string dir = Directory.Exists(modelPathOrDir)
                ? modelPathOrDir
                : Path.GetDirectoryName(modelPathOrDir) ?? ".";
            return Path.Combine(dir, FileName);
        }

        /// <summary>
        /// 读取并校验 kws_meta.json。文件不存在抛 <see cref="FileNotFoundException"/>;
        /// JSON 非法或缺少必要字段抛 <see cref="InvalidDataException"/>。
        /// </summary>
        public static FsmnKwsMetadata Load(string metaFilePath)
        {
            if (string.IsNullOrWhiteSpace(metaFilePath))
                throw new ArgumentException("metaFilePath must not be empty.", nameof(metaFilePath));
            if (!File.Exists(metaFilePath))
                throw new FileNotFoundException(
                    $"kws_meta.json not found: {metaFilePath}. 模型目录必须包含 {FileName} (由导出脚本生成)。",
                    metaFilePath);

            JsonDocument doc;
            try
            {
                using var stream = File.OpenRead(metaFilePath);
                doc = JsonDocument.Parse(stream);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"kws_meta.json is not valid JSON: {metaFilePath}", ex);
            }

            using (doc)
            {
                JsonElement root = doc.RootElement;
                if (root.ValueKind != JsonValueKind.Object)
                    throw new InvalidDataException($"kws_meta.json root must be a JSON object: {metaFilePath}");

                string keyword = ReadKeyword(root, metaFilePath);
                int[] tokenIds = ReadTokenIds(root, metaFilePath);
                string[] oov = ReadOovChars(root);
                var (addShift, rescale) = ReadCmvn(root, metaFilePath);
                return new FsmnKwsMetadata(keyword, tokenIds, oov, addShift, rescale);
            }
        }

        private static string ReadKeyword(JsonElement root, string path)
        {
            if (!root.TryGetProperty("keyword", out JsonElement el) || el.ValueKind != JsonValueKind.String)
                throw new InvalidDataException(
                    $"kws_meta.json missing required string field 'keyword': {path}");
            string? keyword = el.GetString();
            if (string.IsNullOrWhiteSpace(keyword))
                throw new InvalidDataException(
                    $"kws_meta.json field 'keyword' must not be empty: {path}");
            return keyword;
        }

        private static int[] ReadTokenIds(JsonElement root, string path)
        {
            if (!root.TryGetProperty("keyword_token_ids", out JsonElement el) || el.ValueKind != JsonValueKind.Array)
                throw new InvalidDataException(
                    $"kws_meta.json missing required array field 'keyword_token_ids': {path}");
            var ids = new List<int>(el.GetArrayLength());
            foreach (JsonElement item in el.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int id))
                    throw new InvalidDataException(
                        $"kws_meta.json field 'keyword_token_ids' must contain only integers: {path}");
                ids.Add(id);
            }
            if (ids.Count == 0)
                throw new InvalidDataException(
                    $"kws_meta.json field 'keyword_token_ids' must not be empty: {path}");
            return ids.ToArray();
        }

        private static string[] ReadOovChars(JsonElement root)
        {
            if (!root.TryGetProperty("keyword_oov_chars", out JsonElement el) || el.ValueKind != JsonValueKind.Array)
                return Array.Empty<string>();
            var chars = new List<string>(el.GetArrayLength());
            foreach (JsonElement item in el.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.String) continue;
                string? s = item.GetString();
                if (!string.IsNullOrEmpty(s)) chars.Add(s);
            }
            return chars.ToArray();
        }

        private static (float[] addShift, float[] rescale) ReadCmvn(JsonElement root, string path)
        {
            if (!root.TryGetProperty("cmvn", out JsonElement cmvn) || cmvn.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"kws_meta.json field 'cmvn' must be an object: {path}");
            return (ReadCmvnValues(cmvn, "add_shift", path), ReadCmvnValues(cmvn, "rescale", path));
        }

        private static float[] ReadCmvnValues(JsonElement cmvn, string name, string path)
        {
            if (!cmvn.TryGetProperty(name, out JsonElement values) || values.ValueKind != JsonValueKind.Array
                || values.GetArrayLength() != CmvnDimension)
                throw new InvalidDataException(
                    $"kws_meta.json field 'cmvn.{name}' must contain {CmvnDimension} numbers: {path}");

            var result = new float[CmvnDimension];
            int index = 0;
            foreach (JsonElement value in values.EnumerateArray())
            {
                if (value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out float number)
                    || !float.IsFinite(number) || (name == "rescale" && number <= 0f))
                    throw new InvalidDataException(
                        $"kws_meta.json field 'cmvn.{name}[{index}]' must be a finite"
                        + (name == "rescale" ? " positive" : "") + $" number: {path}");
                result[index++] = number;
            }
            return result;
        }
    }
}
