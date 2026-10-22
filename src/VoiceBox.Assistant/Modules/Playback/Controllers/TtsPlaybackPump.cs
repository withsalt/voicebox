using Microsoft.Extensions.Logging;
using System.Text;
using System.Threading.Channels;
using VoiceBox.Assistant.Modules.Playback.Buffering;
using VoiceBox.Assistant.Modules.Playback.Contracts;
using VoiceBox.Assistant.Modules.Playback.Models;
using VoiceBox.Assistant.Modules.Response.Models;
using VoiceBox.Audio.Common.Resampling.Service;

namespace VoiceBox.Assistant.Modules.Playback.Controllers
{
    /// <summary>
    /// 把 (LLM 流 → 文本清洗 → 分句 → TTS 流 → 播放队列) 这一段端到端串起来。
    /// 不持有状态机, 只负责"消费一条 LLM 流, 直到播放结束"。
    /// </summary>
    internal sealed class TtsPlaybackPump
    {
        private readonly ISpeechSynthesisStreamPort _tts;
        private readonly BoundedPlaybackWriter _playbackWriter;
        private readonly PlaybackDrainWaiter _drainWaiter;
        private readonly int _minSentenceChars;
        private readonly int _prerollMs;
        private readonly IAudioResampleService _resampler;   // 由调用方注入并持有生命周期 (跨回合缓存)，本类不 Dispose
        private readonly long? _ownerTurnId;
        private readonly IScheduledSpeechSink _scheduledSpeech;
        private float[] _scratch = new float[4096];           // 转换输出复用缓冲，跨句/块复用以避免逐块分配

        public TtsPlaybackPump(
            ISpeechSynthesisStreamPort tts,
            IBoundedPlaybackPort playback,
            ILogger logger,
            int minSentenceChars,
            int tailBufferMs,
            int prerollMs,
            IAudioResampleService resampler,
            long? ownerTurnId,
            IPlaybackStartedSink playbackStarted,
            IScheduledSpeechSink scheduledSpeech)
        {
            _tts = tts ?? throw new ArgumentNullException(nameof(tts));
            _playbackWriter = new BoundedPlaybackWriter(
                playback,
                logger,
                ownerTurnId,
                playbackStarted);
            _drainWaiter = new PlaybackDrainWaiter(playback, logger, tailBufferMs);
            _minSentenceChars = minSentenceChars;
            _prerollMs = prerollMs;
            _resampler = resampler ?? throw new ArgumentNullException(nameof(resampler));
            _ownerTurnId = ownerTurnId;
            _scheduledSpeech = scheduledSpeech ?? throw new ArgumentNullException(nameof(scheduledSpeech));
        }

        /// <summary>
        /// 消费 LLM 流, 把可见文本累积成句子并依次合成播放。
        /// 输入只包含 Response policy 明确标记为可播报的文本。thinking、tool call 和控制事件
        /// 在进入本边界前已经被隔离。
        /// </summary>
        /// <returns>本轮 assistant 完整可见回复 (原文, 不含 thinking, 不清洗)。</returns>
        public async Task<string> RunAsync(
            IAsyncEnumerable<SpeakableTextDelta> textStream,
            CancellationToken ct)
        {
            using var pipelineCancellation = CancellationTokenSource.CreateLinkedTokenSource(ct);
            CancellationToken pipelineToken = pipelineCancellation.Token;
            var sentences = new SpeakableSentenceQueue(_minSentenceChars);
            var visible = new StringBuilder();
            int ttsOutputSampleRate = _tts.OutputSampleRate;
            int ttsOutputChannels = Math.Max(1, _tts.OutputChannels);
            // 播放格式由注入的转换器决定 (其目标由 VoiceAssistant 按 TTS/选项解析)。
            int playbackSampleRate = _resampler.TargetSampleRate;
            int playbackChannels = _resampler.TargetChannels;

            // 播放预缓冲 (抖动缓冲): 先把本轮开头 PlaybackPrerollMs 的音频攒住, 攒够再一次性灌入
            // 播放队列, 之后转为逐块直通。这样播放一开始就有一段领先缓冲, 合成偶发跟不上 1x 播放
            // 速度 (例如句与句之间的首包等待) 时不会立刻把队列抽干而出现欠载卡顿。
            // 阈值以"播放格式的交织样本数"计; 整轮只在最前面建一次缓冲垫, 之后一直受用。
            var preroll = new PrerollGate(
                ComputePrerollThresholdSamples(playbackSampleRate, playbackChannels),
                _playbackWriter);
            Task<long> synthesisTask = RunSynthesisAsync();

            async Task<long> RunSynthesisAsync()
            {
                try
                {
                    return await SynthesizeQueuedSentencesAsync(
                        sentences.Reader, preroll, playbackChannels,
                        ttsOutputSampleRate, ttsOutputChannels, pipelineToken).ConfigureAwait(false);
                }
                catch
                {
                    // A failed consumer must release a producer awaiting its next model delta.
                    // Keep the synthesis exception on this task so the caller sees the cause.
                    await CancelPipelineAsync().ConfigureAwait(false);
                    throw;
                }
            }

            async Task CancelPipelineAsync()
            {
                // Cleanup callbacks must not replace the original model or synthesis failure.
                try { await pipelineCancellation.CancelAsync().ConfigureAwait(false); }
                catch { }
            }

            try
            {
                await foreach (SpeakableTextDelta delta in textStream.WithCancellation(pipelineToken).ConfigureAwait(false))
                {
                    if (!string.IsNullOrEmpty(delta.Text))
                    {
                        visible.Append(delta.Text);

                        await sentences.WriteAsync(delta.Text, pipelineToken).ConfigureAwait(false);
                        if (delta.ForceSpeechBoundary)
                        {
                            TaskCompletionSource completion = delta.SpeechBoundaryCompletion
                                ?? new TaskCompletionSource(
                                    TaskCreationOptions.RunContinuationsAsynchronously);
                            await sentences.FlushAsync(completion, pipelineToken).ConfigureAwait(false);
                            await AwaitSpeechBoundaryAsync(completion.Task, synthesisTask, pipelineToken)
                                .ConfigureAwait(false);
                        }
                    }
                }

                await sentences.CompleteAsync(pipelineToken).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                sentences.Fail(ex);
                await CancelPipelineAsync().ConfigureAwait(false);
                try { await synthesisTask.ConfigureAwait(false); }
                catch when (ex is not OperationCanceledException || ct.IsCancellationRequested) { }
                throw;
            }

            long enqueuedSamples = await synthesisTask.ConfigureAwait(false);
            await _drainWaiter.WaitAsync(enqueuedSamples, playbackSampleRate, ct).ConfigureAwait(false);
            return visible.ToString();
        }

        /// <summary>
        /// 把 <see cref="_prerollMs"/> 毫秒换算成播放格式下的交织样本数阈值
        /// (= 毫秒 × 播放采样率 × 声道数 / 1000)。&lt;= 0 表示不预缓冲, 行为与旧版逐块直通一致。
        /// </summary>
        private int ComputePrerollThresholdSamples(int sampleRate, int channels)
        {
            if (_prerollMs <= 0 || sampleRate <= 0 || channels <= 0) return 0;
            long samples = (long)_prerollMs * sampleRate * channels / 1000;
            return samples > int.MaxValue ? int.MaxValue : (int)samples;
        }

        private async Task<long> SynthesizeQueuedSentencesAsync(
            ChannelReader<SpeakableSentenceQueue.SentenceItem> sentences,
            PrerollGate preroll,
            int playbackChannels,
            int sourceSampleRateFallback,
            int sourceChannelsFallback,
            CancellationToken ct)
        {
            long enqueuedSamples = 0;
            long lastBoundarySamples = 0;
            await foreach (SpeakableSentenceQueue.SentenceItem item in
                sentences.ReadAllAsync(ct).ConfigureAwait(false))
            {
                try
                {
                    if (item.Text is not null)
                    {
                        enqueuedSamples += await SynthesizeAndEnqueueAsync(
                            item.Text,
                            preroll,
                            playbackChannels,
                            sourceSampleRateFallback,
                            sourceChannelsFallback,
                            ct).ConfigureAwait(false);
                    }

                    if (item.Completion is not null)
                    {
                        if (enqueuedSamples <= lastBoundarySamples)
                        {
                            throw new InvalidOperationException(
                                "A forced speech boundary did not produce any audio samples.");
                        }
                        // A boundary must mean that audio has reached the device playback
                        // queue, not merely the optional preroll buffer. Once released, the
                        // remaining response streams through the already established queue.
                        await preroll.FlushAsync(ct).ConfigureAwait(false);
                        lastBoundarySamples = enqueuedSamples;
                    }
                    item.Completion?.TrySetResult();
                }
                catch (Exception exception)
                {
                    item.Completion?.TrySetException(exception);
                    throw;
                }
            }

            // 整轮音频不足预缓冲阈值时, 把仍攒在缓冲里的尾巴全部送出, 避免短回复一直憋着不播。
            await preroll.FlushAsync(ct).ConfigureAwait(false);
            return enqueuedSamples;
        }

        private async Task<int> SynthesizeAndEnqueueAsync(
            string sentence,
            PrerollGate preroll,
            int playbackChannels,
            int sourceSampleRateFallback,
            int sourceChannelsFallback,
            CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(sentence)) return 0;

            // 每句重置重采样相位 (等价于过去"每句新建转换器")。
            _resampler.Reset();
            int sampleCount = 0;
            bool speechScheduled = false;

            await foreach (SynthesizedAudioChunk ttsChunk in _tts
                .SynthesizeAsync(sentence, ct)
                .ConfigureAwait(false))
            {
                if (ttsChunk.Samples.Length == 0) continue;
                // 自适应: 优先用 chunk 自描述格式, 缺失时回落到 TTS 服务声明的输出格式。
                int srcRate = ttsChunk.SampleRate > 0 ? ttsChunk.SampleRate : sourceSampleRateFallback;
                int srcChannels = ttsChunk.Channels > 0 ? ttsChunk.Channels : sourceChannelsFallback;
                if (!speechScheduled)
                {
                    speechScheduled = true;
                    NotifySpeechScheduled(sentence);
                }
                // 写入复用缓冲 _scratch (按需扩容), 零逐块分配。
                int produced = _resampler.Convert(ttsChunk.Samples, srcRate, srcChannels, ref _scratch);
                if (produced == 0) continue;
                await preroll.EnqueueAsync(_scratch, produced, ct).ConfigureAwait(false);
                sampleCount += produced / playbackChannels;
            }

            // 冲洗重采样器群延迟尾部, 避免句尾被截断。
            int flushed = _resampler.Flush(ref _scratch);
            if (flushed > 0)
            {
                if (!speechScheduled)
                    NotifySpeechScheduled(sentence);
                await preroll.EnqueueAsync(_scratch, flushed, ct).ConfigureAwait(false);
                sampleCount += flushed / playbackChannels;
            }

            return sampleCount;
        }

        private static async Task AwaitSpeechBoundaryAsync(
            Task boundary,
            Task synthesisTask,
            CancellationToken cancellationToken)
        {
            Task completed = await Task.WhenAny(boundary, synthesisTask).ConfigureAwait(false);
            if (ReferenceEquals(completed, synthesisTask))
                await synthesisTask.ConfigureAwait(false);
            await boundary.WaitAsync(cancellationToken).ConfigureAwait(false);
        }

        private void NotifySpeechScheduled(string text)
        {
            if (_ownerTurnId is { } turnId)
                _scheduledSpeech.OnSpeechScheduled(turnId, text);
        }

        /// <summary>
        /// 播放预缓冲门 (抖动缓冲)。开播前先把前 N 个交织样本攒在内存里, 攒够阈值再一次性灌入
        /// 播放队列, 之后转为直通逐块入队。这样播放一开始就握有一段领先缓冲, 后续合成偶发慢于 1x
        /// 播放速度时, 队列不会立即见底而插静音 (听感上的卡顿)。一旦释放就不再二次预缓冲——持续慢于
        /// 实时的情况靠预缓冲也救不回来, 这里只吸收瞬时抖动, 与各 Demo 的 preroll 行为一致。
        /// 每个回合 (一次 <see cref="RunAsync"/>) 新建一个实例, 不跨回合复用。
        /// </summary>
        private sealed class PrerollGate
        {
            private readonly int _thresholdSamples;
            private readonly BoundedPlaybackWriter _playbackWriter;
            private readonly List<float[]> _buffered = new();
            private int _bufferedSamples;
            private bool _released;

            public PrerollGate(int thresholdSamples, BoundedPlaybackWriter playbackWriter)
            {
                _thresholdSamples = thresholdSamples;
                _playbackWriter = playbackWriter ?? throw new ArgumentNullException(nameof(playbackWriter));
                _released = thresholdSamples <= 0;   // 阈值 <=0: 不预缓冲, 直接直通 (等价旧行为)
            }

            /// <summary>
            /// 入队一段样本。仍在预缓冲阶段时先拷贝攒住 (注意 <paramref name="buffer"/> 是上层复用的
            /// _scratch, 下一块会覆盖它, 必须拷贝), 攒够阈值即整体释放; 已释放则直接直通入队。
            /// </summary>
            public async Task EnqueueAsync(float[] buffer, int count, CancellationToken ct)
            {
                if (count <= 0) return;

                if (_released)
                {
                    await _playbackWriter.EnqueueAsync(buffer, count, ct).ConfigureAwait(false);
                    return;
                }

                _buffered.Add(buffer.AsSpan(0, count).ToArray());
                _bufferedSamples += count;
                if (_bufferedSamples >= _thresholdSamples)
                    await ReleaseAsync(ct).ConfigureAwait(false);
            }

            /// <summary>把仍攒在预缓冲里的内容全部送出。用于整轮音频不足阈值时的收尾。</summary>
            public async Task FlushAsync(CancellationToken ct)
            {
                if (!_released)
                    await ReleaseAsync(ct).ConfigureAwait(false);
            }

            private async Task ReleaseAsync(CancellationToken ct)
            {
                _released = true;
                foreach (float[] block in _buffered)
                    await _playbackWriter.EnqueueAsync(block, block.Length, ct).ConfigureAwait(false);
                _buffered.Clear();
                _bufferedSamples = 0;
            }
        }
    }
}

