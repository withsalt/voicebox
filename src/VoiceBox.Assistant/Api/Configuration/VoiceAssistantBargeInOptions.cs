namespace VoiceBox.Assistant.Api.Configuration;

/// <summary>
/// 语音助手可打断回复 (barge-in) 参数。
/// </summary>
public sealed class VoiceAssistantBargeInOptions
{
    /// <summary>
    /// 是否允许在助手回复时打断。关闭说话人确认时，DTD 可直接停播；开启时必须由当前候选的
    /// 说话人匹配授权。停播后由整句识别结果决定是否开始新回合。建议配合 AEC/DNR 使用，减少
    /// TTS 回声造成的误触发。
    /// </summary>
    public bool Enable { get; set; } = false;

    /// <summary>打断前需要累计的候选语音时长，用于过滤短促误报。0 使用默认值；默认 180 ms。</summary>
    public int MinDurationMs { get; set; } = 180;

    /// <summary>
    /// 近端检测使用的最低 RMS 参考。设为 0 可关闭固定参考；默认 0.03（约 -30 dBFS）。
    /// </summary>
    public float MinRms { get; set; } = 0.03f;

    /// <summary>
    /// 播放期间忽略候选开头的确认时长，用于过滤回声和短促误报。确认后这段音频仍会补喂 ASR。
    /// 0 表示关闭；默认 120 ms。
    /// </summary>
    public int PlaybackCandidateSuppressionMs { get; set; } = 120;

    /// <summary>
    /// 候选未确认时允许累计的最长时长。超时后重开候选并丢弃旧音频；0 表示不限制，默认 1200 ms。
    /// </summary>
    public int PlaybackMaxCandidateMs { get; set; } = 1200;

    /// <summary>
    /// 确认打断后补喂给 ASR 的候选音频上限。0 表示不限制；默认 1000 ms。
    /// </summary>
    public int PlaybackRecognitionAudioLimitMs { get; set; } = 1000;

    /// <summary>
    /// TTS 开始播放后的打断保护时长，给 AEC 留出稳定时间。期间仍缓存音频，超时后可作为 ASR 预滚。
    /// 0 表示关闭；默认 500 ms。
    /// </summary>
    public int PlaybackStartupGuardMs { get; set; } = 500;

    /// <summary>
    /// 未开启声纹认证时，初步近端语音证据触发的临时播放增益。DTD 确认后停播，误报则恢复原增益。
    /// 范围 [0, 1]；1 关闭 ducking，默认 0.15。开启声纹认证时忽略。
    /// </summary>
    public float PlaybackDuckGain { get; set; } = 0.15f;

    /// <summary>临时降音量后，连续多久没有新近端语音证据就恢复原音量。默认 200 ms。</summary>
    public int PlaybackDuckReleaseMs { get; set; } = 200;

    /// <summary>主动近端检测的判定窗口长度。默认 360 ms。</summary>
    public int ActiveNearEndWindowMs { get; set; } = 360;

    /// <summary>
    /// 确认后在语音起点前补给 ASR 的音频时长，取自处理后的采集流并与判定窗口对齐。默认 360 ms。
    /// </summary>
    public int ActiveNearEndPreRollMs { get; set; } = 360;

    /// <summary>近端检测多久没有新语音后合成结束事件；若 VAD 先结束则以 VAD 为准。默认 520 ms。</summary>
    public int ActiveNearEndEndSilenceMs { get; set; } = 520;

    /// <summary>
    /// 语音结束后等待 AEC/NS/DNR 处理链补齐尾音的最长时间，防止漏掉句尾或一直等待。默认 1000 ms。
    /// </summary>
    public int ActiveNearEndAudioDrainTimeoutMs { get; set; } = 1000;

    /// <summary>一次主动打断结束后暂停检测的时长，用于避免语气词或残响立即再次触发。默认 800 ms。</summary>
    public int ActiveNearEndCooldownMs { get; set; } = 800;

    /// <summary>Render-aware double-talk detector settings used while playback is active.</summary>
    public VoiceAssistantRenderAwareDtdOptions Dtd { get; set; } = new();

    /// <summary>Render/capture timeline alignment settings for the double-talk detector.</summary>
    public VoiceAssistantRenderAlignmentOptions RenderAlignment { get; set; } = new();
}
