using VoiceBox.Core.Services.WakeupService.Models;

namespace VoiceBox.Core.Services.WakeupService
{
    public interface IWakeupService : IDisposable
    {
        event EventHandler<WakeupEventArgs>? OnWakeup;

        void Start();

        void Stop();

        /// <summary>
        /// 动态设置唤醒词触发阈值。
        /// </summary>
        /// <param name="threshold">触发阈值, 范围 (0, 1]。</param>
        /// <remarks>
        /// 内置唤醒服务支持在不重新加载模型的情况下更新阈值。
        /// 对包含多个唤醒词的服务, 该值作用于全部唤醒词。
        /// 未实现此能力的第三方服务会抛出 <see cref="NotSupportedException"/>。
        /// </remarks>
        void SetThreshold(float threshold) =>
            throw new NotSupportedException("Dynamic wakeup threshold is not supported by this service.");

        void FeedAudio(ReadOnlySpan<float> audioData);

        bool IsListening { get; }
    }
}
