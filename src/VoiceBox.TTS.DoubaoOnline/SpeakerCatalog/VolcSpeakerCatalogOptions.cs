namespace VoiceBox.TTS.DoubaoOnline.SpeakerCatalog
{
    /// <summary>
    /// <see cref="VolcSpeakerCatalogClient"/> 的配置。<c>ListSpeakers</c> 属于管理类 OpenAPI，
    /// 使用火山引擎签名 V4 (AccessKeyId + SecretAccessKey)，与合成用的 <c>X-Api-Key</c> 是<b>不同</b>的凭证。
    /// AK/SK 可在「火山引擎控制台 &gt; 访问控制 &gt; 密钥管理」获取。
    /// </summary>
    public sealed class VolcSpeakerCatalogOptions
    {
        /// <summary>访问密钥 ID (Access Key ID)。</summary>
        public string AccessKeyId { get; set; } = string.Empty;

        /// <summary>私有访问密钥 (Secret Access Key)。</summary>
        public string SecretAccessKey { get; set; } = string.Empty;

        /// <summary>OpenAPI 域名。</summary>
        public string Host { get; set; } = "open.volcengineapi.com";

        /// <summary>区域，固定 cn-beijing。</summary>
        public string Region { get; set; } = "cn-beijing";

        /// <summary>服务代号，固定 speech_saas_prod。</summary>
        public string Service { get; set; } = "speech_saas_prod";

        /// <summary>接口版本 (Version)。</summary>
        public string Version { get; set; } = "2025-05-20";

        /// <summary>请求超时 (秒)。</summary>
        public int TimeoutSec { get; set; } = 15;
    }
}
