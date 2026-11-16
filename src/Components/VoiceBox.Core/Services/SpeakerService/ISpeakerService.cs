using VoiceBox.Core.Services.SpeakerService.Models;

namespace VoiceBox.Core.Services.SpeakerService
{
    /// <summary>
    /// 说话人验证 (声纹) 服务。
    /// <para>
    /// 提供三类能力:
    /// 1. 离线 — 从整段音频提取说话人 embedding 并计算两段音频的相似度得分;
    /// 2. 注册 — 用一段音频注册说话人档案 (声纹库);
    /// 3. 流式 — 持续喂入实时音频, 按滑动窗口周期性产出与已注册说话人的相似度结果
    ///    (用于实时语音对话中的目标说话人检测)。
    /// </para>
    /// </summary>
    public interface ISpeakerService : IDisposable
    {
        /// <summary>说话人 embedding 维度。</summary>
        int EmbeddingSize { get; }

        /// <summary>服务要求的输入采样率 (Hz)。所有音频入口均须使用该采样率。</summary>
        int SampleRate { get; }

        /// <summary>判定阈值 (余弦相似度, 得分大于该值视为同一说话人)。</summary>
        float Threshold { get; }

        /// <summary>
        /// 流式结果实际覆盖的音频样本数。该值必须描述特征前端真实使用的样本跨度，
        /// 不能仅返回名义配置时长；例如 snip-edges 特征窗可能使两者相差若干毫秒。
        /// </summary>
        int StreamingWindowSamples { get; }

        /// <summary>相邻流式结果结束点之间的实际样本步长。</summary>
        int StreamingHopSamples { get; }

        /// <summary>
        /// 提取整段音频的说话人 embedding (L2 归一化)。
        /// 输入: 单声道 float32 [-1, 1], 采样率须与服务配置一致。
        /// </summary>
        float[] ExtractEmbedding(ReadOnlySpan<float> samples);

        /// <summary>计算两个 embedding 的余弦相似度 [-1, 1]。</summary>
        float CompareEmbeddings(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2);

        /// <summary>计算两段音频的说话人相似度得分 (余弦)。</summary>
        float Verify(ReadOnlySpan<float> samples1, ReadOnlySpan<float> samples2);

        /// <summary>
        /// 注册 (或覆盖) 一个说话人档案。建议提供 3~10 秒有效语音。
        /// </summary>
        void Enroll(string speakerName, ReadOnlySpan<float> samples);

        /// <summary>移除说话人档案。返回是否存在并移除成功。</summary>
        bool Unenroll(string speakerName);

        /// <summary>已注册的说话人名称。</summary>
        IReadOnlyCollection<string> EnrolledSpeakers { get; }

        /// <summary>
        /// 流式检测结果 (滑动窗口产出)。
        /// 注意: 实现可跳过部分窗口不产出结果 (如静音门); 无注册说话人时
        /// <see cref="Models.SpeakerEventArgs.BestSpeaker"/> 为 null (仅携带 embedding)。
        /// 实现可选支持前缀窗提前打分: 首个完整窗就绪前额外产出
        /// <see cref="Models.SpeakerEventArgs.IsEarlyExit"/> 为 true 的临时结果。
        /// </summary>
        event EventHandler<SpeakerEventArgs>? OnResult;

        /// <summary>喂入实时音频 (单声道 float32 [-1, 1])。积累到完整窗口时触发 <see cref="OnResult"/>。</summary>
        void FeedAudio(ReadOnlySpan<float> samples);

        /// <summary>重置流式状态 (清空音频缓冲与平滑得分, 保留已注册说话人)。</summary>
        void Reset();
    }
}
