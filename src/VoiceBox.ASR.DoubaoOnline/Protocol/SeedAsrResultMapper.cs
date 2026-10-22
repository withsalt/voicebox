using System.Text;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.ASR.DoubaoOnline.Protocol
{
    /// <summary>
    /// 将服务端 <see cref="ResponsePayload"/> 映射为 Core 的 <see cref="AsrResult"/>。
    /// <para>
    /// 语义约定 (与 <see cref="AsrResult.CommittedText"/> / <see cref="AsrResult.UncommittedText"/> 对齐)：
    ///   * <c>Text</c>：服务端给出的整段识别文本 (result.text)。
    ///   * <c>CommittedText</c>：已确定 (definite=true) 分句拼接而成的稳定前缀。
    ///   * <c>UncommittedText</c>：尾部尚未确定 (definite=false) 分句，可能被后续轮次修订。
    /// </para>
    /// 当未开启 <c>show_utterances</c> 时无法区分已/未确定，Committed/Uncommitted 置为 null。
    /// </summary>
    internal static class SeedAsrResultMapper
    {
        public static AsrResult Map(ResponsePayload payload, string language)
        {
            var result = new AsrResult { Language = language ?? string.Empty };

            var body = payload.Result;
            if (body is null)
                return result;

            var utterances = body.Utterances;
            if (utterances is { Count: > 0 })
            {
                var committed = new StringBuilder();
                var uncommitted = new StringBuilder();
                foreach (var u in utterances)
                {
                    if (string.IsNullOrEmpty(u.Text)) continue;
                    (u.Definite ? committed : uncommitted).Append(u.Text);
                }

                result.CommittedText = committed.ToString();
                result.UncommittedText = uncommitted.ToString();
                // Prefer the explicit full text; fall back to concatenated utterances.
                result.Text = !string.IsNullOrEmpty(body.Text)
                    ? body.Text!
                    : committed.ToString() + uncommitted.ToString();
            }
            else
            {
                result.Text = body.Text ?? string.Empty;
            }

            return result;
        }
    }
}
