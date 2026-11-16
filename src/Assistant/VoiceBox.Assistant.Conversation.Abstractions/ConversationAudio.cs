namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 一段独立持有的 Float32 PCM 音频快照。多声道采样按帧交错排列，不包含文件头。
/// </summary>
public sealed class ConversationAudio
{
    /// <summary>复制采样数据，调用方随后可以释放或复用输入缓冲区。</summary>
    public ConversationAudio(ReadOnlySpan<float> samples, int sampleRate, int channels = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        if (samples.IsEmpty)
            throw new ArgumentException("Audio samples must not be empty.", nameof(samples));
        if (samples.Length % channels != 0)
            throw new ArgumentException("Audio samples must contain complete interleaved frames.", nameof(samples));

        Samples = samples.ToArray();
        SampleRate = sampleRate;
        Channels = channels;
    }

    /// <summary>Float32 PCM 采样；会话可以在请求结束后继续持有该快照。</summary>
    public ReadOnlyMemory<float> Samples { get; }

    /// <summary>每个声道每秒的采样帧数。</summary>
    public int SampleRate { get; }

    /// <summary>声道数。多声道时 Samples 按帧交错排列。</summary>
    public int Channels { get; }

    /// <summary>由采样帧数和采样率计算的音频时长。</summary>
    public TimeSpan Duration => TimeSpan.FromSeconds(Samples.Length / (double)Channels / SampleRate);
}
