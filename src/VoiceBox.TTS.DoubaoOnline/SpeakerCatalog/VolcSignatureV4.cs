using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace VoiceBox.TTS.DoubaoOnline.SpeakerCatalog
{
    /// <summary>
    /// 火山引擎 OpenAPI 签名 V4 (HMAC-SHA256)。用于 <see cref="VolcSpeakerCatalogClient"/> 调用
    /// <c>ListSpeakers</c> 等管理类接口。与 AWS SigV4 类似，但签名密钥派生<b>不加</b> "AWS4" 前缀，
    /// 且凭证串终止段为 <c>request</c>。
    /// </summary>
    /// <remarks>
    /// 纯函数、可注入时间戳，便于用已知向量做确定性单测。产出的头包含 <c>X-Date</c>、
    /// <c>X-Content-Sha256</c>、<c>Authorization</c>；调用方再补 <c>Host</c>/<c>Content-Type</c>。
    /// </remarks>
    internal static class VolcSignatureV4
    {
        private const string Algorithm = "HMAC-SHA256";

        public sealed record Request(
            string Method,
            string Host,
            string Path,
            IReadOnlyList<KeyValuePair<string, string>> Query,
            byte[] Body,
            string AccessKeyId,
            string SecretAccessKey,
            string Region,
            string Service,
            DateTime UtcNow);

        /// <summary>计算并返回签名所需的请求头 (X-Date / X-Content-Sha256 / Authorization)。</summary>
        public static Dictionary<string, string> Sign(Request req)
        {
            string xDate = req.UtcNow.ToString("yyyyMMddTHHmmssZ", CultureInfo.InvariantCulture);
            string shortDate = xDate[..8];
            string payloadHash = HexSha256(req.Body);

            string canonicalQuery = BuildCanonicalQuery(req.Query);

            // 签名头固定为 host;x-content-sha256;x-date (与官方示例一致)。
            const string signedHeaders = "host;x-content-sha256;x-date";
            string canonicalHeaders =
                $"host:{req.Host}\n" +
                $"x-content-sha256:{payloadHash}\n" +
                $"x-date:{xDate}\n";

            string canonicalRequest = string.Join('\n',
                req.Method.ToUpperInvariant(),
                string.IsNullOrEmpty(req.Path) ? "/" : req.Path,
                canonicalQuery,
                canonicalHeaders,
                signedHeaders,
                payloadHash);

            string credentialScope = $"{shortDate}/{req.Region}/{req.Service}/request";
            string stringToSign = string.Join('\n',
                Algorithm,
                xDate,
                credentialScope,
                HexSha256(Encoding.UTF8.GetBytes(canonicalRequest)));

            byte[] signingKey = DeriveSigningKey(req.SecretAccessKey, shortDate, req.Region, req.Service);
            string signature = ToHex(HmacSha256(signingKey, Encoding.UTF8.GetBytes(stringToSign)));

            string authorization =
                $"{Algorithm} Credential={req.AccessKeyId}/{credentialScope}, " +
                $"SignedHeaders={signedHeaders}, Signature={signature}";

            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["X-Date"] = xDate,
                ["X-Content-Sha256"] = payloadHash,
                ["Authorization"] = authorization,
            };
        }

        private static string BuildCanonicalQuery(IReadOnlyList<KeyValuePair<string, string>> query)
        {
            if (query.Count == 0) return string.Empty;
            // 按编码后的 key 升序 (相同 key 再按 value)，key/value 均做 RFC3986 编码。
            var encoded = query
                .Select(kv => (K: UriEncode(kv.Key), V: UriEncode(kv.Value)))
                .OrderBy(p => p.K, StringComparer.Ordinal)
                .ThenBy(p => p.V, StringComparer.Ordinal);
            return string.Join('&', encoded.Select(p => $"{p.K}={p.V}"));
        }

        private static byte[] DeriveSigningKey(string secret, string shortDate, string region, string service)
        {
            // 火山引擎: 直接以 SecretAccessKey 为初始密钥 (无 "AWS4" 前缀)。
            byte[] kDate = HmacSha256(Encoding.UTF8.GetBytes(secret), Encoding.UTF8.GetBytes(shortDate));
            byte[] kRegion = HmacSha256(kDate, Encoding.UTF8.GetBytes(region));
            byte[] kService = HmacSha256(kRegion, Encoding.UTF8.GetBytes(service));
            return HmacSha256(kService, Encoding.UTF8.GetBytes("request"));
        }

        private static byte[] HmacSha256(byte[] key, byte[] data)
        {
            using var h = new HMACSHA256(key);
            return h.ComputeHash(data);
        }

        private static string HexSha256(byte[] data) => ToHex(SHA256.HashData(data));

        private static string ToHex(byte[] bytes) => Convert.ToHexStringLower(bytes);

        /// <summary>RFC3986 编码 (与 AWS/火山一致：不编码 A-Z a-z 0-9 - _ . ~)。</summary>
        internal static string UriEncode(string value)
        {
            var sb = new StringBuilder(value.Length * 2);
            foreach (byte b in Encoding.UTF8.GetBytes(value))
            {
                char c = (char)b;
                if (c is (>= 'A' and <= 'Z') or (>= 'a' and <= 'z') or (>= '0' and <= '9')
                    or '-' or '_' or '.' or '~')
                    sb.Append(c);
                else
                    sb.Append('%').Append(b.ToString("X2", CultureInfo.InvariantCulture));
            }
            return sb.ToString();
        }
    }
}
