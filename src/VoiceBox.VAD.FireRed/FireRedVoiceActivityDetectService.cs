using System.Diagnostics;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;
using VoiceBox.VAD.FireRed.Models;

namespace VoiceBox.VAD.FireRed
{
    public sealed class FireRedVoiceActivityDetectService : IVoiceActivityDetectService
    {
        public event EventHandler<EventArgs>? OnSpeechStarted;
        public event EventHandler<VADEventArgs>? OnSpeaking;
        public event EventHandler<VADEventArgs>? OnSpeechEnded;
        public event EventHandler<VadProbabilityEventArgs>? OnProbability;

        private readonly FireRedVadOptions _options;
        private readonly KaldiFbankExtractor _fbank;
        private readonly float[] _cmvnMeans;
        private readonly float[] _cmvnInvStd;
        private FireRedVadOnnxRunner? _onnx;

        private readonly object _lock = new();
        private readonly Stopwatch _stopwatch = new();
        private bool _disposed;

        // 状态机
        private enum VadState { Silence, PossibleSpeech, Speech, PossibleSilence }
        private VadState _state = VadState.Silence;
        // 帧号/帧计数器用 long: 每帧 10ms, int 在约 248 天连续运行(无 Reset)后会溢出转负,
        // 导致环形缓冲下标 ((globalFrameIdx-1) % capacity) 变负而越界崩溃。
        // _speechCnt 不在此列: 它在状态切换/MaxSpeechFrame 处频繁清零, 有界, 保持 int。
        private long _frameCnt;
        private readonly Queue<float> _smoothWindow = new();
        private float _smoothSum;
        private long _probabilityFrameCnt;
        private readonly Queue<float> _probabilitySmoothWindow = new();
        private float _probabilitySmoothSum;
        private int _speechCnt;
        private long _silenceCnt;
        private bool _hitMaxSpeech;
        private long _lastSpeechStartFrame = -1;
        private long _lastSpeechEndFrame = -1;

        // 帧音频历史（用于回溯前置 padding）：扁平环形缓冲，每帧 FrameShift 个 [-1,1] 量级采样。
        // _frameAudioRingWriteFrame 表示下一个将要写入的全局帧号（1-based），即“已写入帧数 + 1”。
        private readonly float[] _frameAudioRing;
        private readonly int _frameAudioRingCapacity; // 帧数 (固定容量, 保持 int)
        private long _frameAudioRingWriteFrame = 1;

        // 检测到说话之前的原始音频环形缓冲（[-1,1] 量级）
        private readonly RingBufferQueue<float>? _preSpeechBuffer;
        // Predict 累计输入的样本总数（用于将 _preSpeechBuffer 与帧序号对齐）
        private long _audioInputSampleCount;

        // fbank 跨调用样本缓冲（int16 量级 float）
        private float[] _fbankRemainder = Array.Empty<float>();

        // 当前正在发射的语音段
        private bool _isEmittingSpeech;
        private readonly List<float> _speechBuffer;
        private long _lastEmittedFrame;

        private enum VadEventType { SpeechStart, SpeechEnd }
        private readonly struct VadEvent
        {
            public VadEvent(VadEventType type, long startFrame, long endFrame)
            {
                Type = type;
                StartFrame = startFrame;
                EndFrame = endFrame;
            }
            public VadEventType Type { get; }
            public long StartFrame { get; }
            public long EndFrame { get; }
        }

        private readonly record struct VadProbabilityObservation(
            float Probability,
            float SmoothedProbability,
            bool IsSpeech,
            long FrameIndex);

        public FireRedVoiceActivityDetectService(IOptions<FireRedVadOptions> options)
        {
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));

            if (_options.SampleRate != KaldiFbankExtractor.SampleRate)
                throw new ArgumentException($"FireRedVAD only supports {KaldiFbankExtractor.SampleRate}Hz, got {_options.SampleRate}.");

            if (string.IsNullOrWhiteSpace(_options.VADModelPath))
                throw new ArgumentNullException(nameof(_options.VADModelPath), "VADModelPath cannot be null or empty.");
            if (!File.Exists(_options.VADModelPath))
                throw new FileNotFoundException($"VAD Model file not found: {_options.VADModelPath}");

            if (string.IsNullOrWhiteSpace(_options.CMVNPath))
                throw new ArgumentNullException(nameof(_options.CMVNPath), "CMVNPath cannot be null or empty.");
            if (!File.Exists(_options.CMVNPath))
                throw new FileNotFoundException($"CMVN file not found: {_options.CMVNPath}");

            // pad_start_frame 至少为 smooth_window_size，与 python 行为一致
            if (_options.PadStartFrame < _options.SmoothWindowSize)
                _options.PadStartFrame = _options.SmoothWindowSize;

            _fbank = new KaldiFbankExtractor();
            (_cmvnMeans, _cmvnInvStd) = KaldiCmvnReader.Load(_options.CMVNPath);
            if (_cmvnMeans.Length != KaldiFbankExtractor.NumBins)
                throw new InvalidDataException($"CMVN dim {_cmvnMeans.Length} does not match fbank dim {KaldiFbankExtractor.NumBins}.");

            _onnx = new FireRedVadOnnxRunner(_options.VADModelPath, _options);

            // 环形缓冲容量需覆盖最大可能回溯跨度：MaxSpeechFrame（强制结束时回放整段）
            // 加上 PadStartFrame（语音起点回溯）以及一些余量。
            _frameAudioRingCapacity = Math.Max(64,
                _options.MaxSpeechFrame + _options.PadStartFrame + _options.SmoothWindowSize + _options.MinSpeechFrame + 16);
            _frameAudioRing = new float[_frameAudioRingCapacity * KaldiFbankExtractor.FrameShift];

            if (_options.PreSpeechPaddingMs > 0)
            {
                int preSpeechPaddingSamples = (int)(_options.SampleRate * (_options.PreSpeechPaddingMs / 1000f));
                if (preSpeechPaddingSamples > 0)
                {
                    _preSpeechBuffer = new RingBufferQueue<float>(preSpeechPaddingSamples);
                }
            }

            _speechBuffer = new List<float>(KaldiFbankExtractor.SampleRate * 10);
        }

        public void Predict(ReadOnlySpan<float> audioData)
        {
            lock (_lock)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(FireRedVoiceActivityDetectService));
                if (audioData.IsEmpty)
                    return;

                bool hasSpeechSubscribers = OnSpeaking != null || OnSpeechStarted != null || OnSpeechEnded != null;
                bool hasProbabilitySubscribers = OnProbability != null;
                if (!hasSpeechSubscribers && !hasProbabilitySubscribers)
                    return;

                if (hasSpeechSubscribers)
                {
                    // 0) 在进入 fbank 之前，先把原始 [-1,1] 音频写入前置语音环形缓冲，
                    //    并累计输入样本计数（与帧序号对齐用）。
                    _preSpeechBuffer?.EnqueueRange(audioData);
                    _audioInputSampleCount += audioData.Length;
                }

                // 1) 拼接 fbank 剩余样本，并将 [-1,1] 缩放到 int16 量级
                int totalLen = _fbankRemainder.Length + audioData.Length;
                var combined = new float[totalLen];
                _fbankRemainder.AsSpan().CopyTo(combined.AsSpan());
                for (int i = 0; i < audioData.Length; i++)
                {
                    combined[_fbankRemainder.Length + i] = audioData[i] * 32768f;
                }

                // 2) FBank 提取
                var fbankFrames = _fbank.Extract(combined, out int consumedSamples);
                int numFrames = fbankFrames.Length;
                if (numFrames == 0)
                {
                    _fbankRemainder = combined;
                    return;
                }

                // 3) 保存剩余样本
                int remainderLen = totalLen - consumedSamples;
                if (remainderLen > 0)
                {
                    var rem = new float[remainderLen];
                    Buffer.BlockCopy(combined, consumedSamples * sizeof(float), rem, 0, remainderLen * sizeof(float));
                    _fbankRemainder = rem;
                }
                else
                {
                    _fbankRemainder = Array.Empty<float>();
                }

                // 4) 应用 CMVN：(fbank - means) * inv_std
                for (int i = 0; i < numFrames; i++)
                {
                    var row = fbankFrames[i];
                    for (int j = 0; j < row.Length; j++)
                    {
                        row[j] = (row[j] - _cmvnMeans[j]) * _cmvnInvStd[j];
                    }
                }

                // 5) 一次性 ONNX 推理，得到每帧的语音概率
                if (_onnx == null)
                    throw new InvalidOperationException("VAD ONNX runner is not initialized.");
                float[] probs = _onnx.Run(fbankFrames);
                if (probs.Length != numFrames)
                {
                    // 兜底：若模型返回不同长度（如 [1,T]），按尾部对齐
                    var aligned = new float[numFrames];
                    int copyLen = Math.Min(numFrames, probs.Length);
                    Array.Copy(probs, probs.Length - copyLen, aligned, numFrames - copyLen, copyLen);
                    probs = aligned;
                }

                // 6) 逐帧驱动状态机并产生事件
                long frameAudioBaseIdx = _frameCnt + 1; // 本次 chunk 中第一帧的全局索引（1-based）
                for (int i = 0; i < numFrames; i++)
                {
                    if (hasSpeechSubscribers)
                    {
                        long globalFrameIdx = frameAudioBaseIdx + i;

                        // 直接把该帧 FrameShift 个采样写入环形缓冲（一边除回 [-1,1] 量级），避免每帧 new float[160]
                        // slot 必落在 [0, capacity) 内, 转回 int 安全。
                        int slot = (int)((globalFrameIdx - 1) % _frameAudioRingCapacity);
                        int srcOffset = i * KaldiFbankExtractor.FrameShift;
                        int destOffset = slot * KaldiFbankExtractor.FrameShift;
                        for (int s = 0; s < KaldiFbankExtractor.FrameShift; s++)
                        {
                            _frameAudioRing[destOffset + s] = combined[srcOffset + s] / 32768f;
                        }
                        _frameAudioRingWriteFrame = globalFrameIdx + 1;

                        var evt = ProcessFrame(probs[i], out VadProbabilityObservation observation);
                        RaiseProbability(observation);
                        if (evt.HasValue)
                        {
                            HandleEvent(evt.Value);
                        }
                    }
                    else
                    {
                        RaiseProbability(ObserveProbabilityOnly(probs[i]));
                    }
                }

                // 7) chunk 结束时，若仍处于发射中，把本 chunk 内未发射的语音音频补发一次
                if (hasSpeechSubscribers && _isEmittingSpeech && _lastEmittedFrame < _frameCnt)
                {
                    var midAudio = ConcatFrameAudio(_lastEmittedFrame + 1, _frameCnt);
                    if (midAudio.Length > 0)
                    {
                        _speechBuffer.AddRange(midAudio);
                        SafeInvoke(OnSpeaking, new VADEventArgs(midAudio, VADEventDataType.Chunk));
                    }
                    _lastEmittedFrame = _frameCnt;
                }
            }
        }

        private VadEvent? ProcessFrame(float rawProb, out VadProbabilityObservation observation)
        {
            observation = ObserveProbability(rawProb);
            bool isSpeech = observation.IsSpeech;

            VadEvent? evt = null;

            // 上一次 max_speech 触发后，本帧需要先发一个 speech_start
            if (_hitMaxSpeech)
            {
                evt = new VadEvent(VadEventType.SpeechStart, _frameCnt, _frameCnt);
                _lastSpeechStartFrame = _frameCnt;
                _hitMaxSpeech = false;
            }

            switch (_state)
            {
                case VadState.Silence:
                    if (isSpeech)
                    {
                        _state = VadState.PossibleSpeech;
                        _speechCnt = 1;
                    }
                    else
                    {
                        _silenceCnt++;
                        _speechCnt = 0;
                    }
                    break;

                case VadState.PossibleSpeech:
                    if (isSpeech)
                    {
                        _speechCnt++;
                        if (_speechCnt >= _options.MinSpeechFrame)
                        {
                            _state = VadState.Speech;
                            long start = Math.Max(1L, _frameCnt - _speechCnt + 1 - _options.PadStartFrame);
                            if (_lastSpeechEndFrame + 1 > start) start = _lastSpeechEndFrame + 1;
                            _lastSpeechStartFrame = start;
                            _silenceCnt = 0;
                            evt = new VadEvent(VadEventType.SpeechStart, start, _frameCnt);
                        }
                    }
                    else
                    {
                        _state = VadState.Silence;
                        _silenceCnt = 1;
                        _speechCnt = 0;
                    }
                    break;

                case VadState.Speech:
                    _speechCnt++;
                    if (isSpeech)
                    {
                        _silenceCnt = 0;
                        if (_speechCnt >= _options.MaxSpeechFrame)
                        {
                            _hitMaxSpeech = true;
                            _speechCnt = 0;
                            evt = new VadEvent(VadEventType.SpeechEnd, _lastSpeechStartFrame, _frameCnt);
                            _lastSpeechEndFrame = _frameCnt;
                            _lastSpeechStartFrame = -1;
                        }
                    }
                    else
                    {
                        _state = VadState.PossibleSilence;
                        _silenceCnt = 1;
                    }
                    break;

                case VadState.PossibleSilence:
                    _speechCnt++;
                    if (isSpeech)
                    {
                        _state = VadState.Speech;
                        _silenceCnt = 0;
                        if (_speechCnt >= _options.MaxSpeechFrame)
                        {
                            _hitMaxSpeech = true;
                            _speechCnt = 0;
                            evt = new VadEvent(VadEventType.SpeechEnd, _lastSpeechStartFrame, _frameCnt);
                            _lastSpeechEndFrame = _frameCnt;
                            _lastSpeechStartFrame = -1;
                        }
                    }
                    else
                    {
                        _silenceCnt++;
                        if (_silenceCnt >= _options.MinSilenceFrame)
                        {
                            _state = VadState.Silence;
                            evt = new VadEvent(VadEventType.SpeechEnd, _lastSpeechStartFrame, _frameCnt);
                            _lastSpeechEndFrame = _frameCnt;
                            _lastSpeechStartFrame = -1;
                            _speechCnt = 0;
                        }
                    }
                    break;
            }

            return evt;
        }

        private VadProbabilityObservation ObserveProbability(float rawProb)
        {
            _frameCnt++;

            if (!float.IsFinite(rawProb))
                rawProb = 0f;
            rawProb = Math.Clamp(rawProb, 0f, 1f);

            // 平滑（滑动平均）
            _smoothWindow.Enqueue(rawProb);
            _smoothSum += rawProb;
            if (_smoothWindow.Count > _options.SmoothWindowSize)
            {
                _smoothSum -= _smoothWindow.Dequeue();
            }

            float smoothed = _smoothSum / _smoothWindow.Count;
            bool isSpeech = smoothed >= _options.SpeechThreshold;
            return new VadProbabilityObservation(rawProb, smoothed, isSpeech, NextProbabilityFrameIndex());
        }

        private VadProbabilityObservation ObserveProbabilityOnly(float rawProb)
        {
            if (!float.IsFinite(rawProb))
                rawProb = 0f;
            rawProb = Math.Clamp(rawProb, 0f, 1f);

            _probabilitySmoothWindow.Enqueue(rawProb);
            _probabilitySmoothSum += rawProb;
            if (_probabilitySmoothWindow.Count > _options.SmoothWindowSize)
            {
                _probabilitySmoothSum -= _probabilitySmoothWindow.Dequeue();
            }

            float smoothed = _probabilitySmoothSum / _probabilitySmoothWindow.Count;
            bool isSpeech = smoothed >= _options.SpeechThreshold;
            return new VadProbabilityObservation(rawProb, smoothed, isSpeech, NextProbabilityFrameIndex());
        }

        private long NextProbabilityFrameIndex()
        {
            _probabilityFrameCnt++;
            return _probabilityFrameCnt;
        }

        private void RaiseProbability(VadProbabilityObservation observation)
        {
            EventHandler<VadProbabilityEventArgs>? handler = OnProbability;
            if (handler == null)
                return;

            var timestamp = TimeSpan.FromMilliseconds((observation.FrameIndex - 1) * KaldiFbankExtractor.FrameShiftMs);
            var frameDuration = TimeSpan.FromMilliseconds(KaldiFbankExtractor.FrameShiftMs);
            SafeInvoke(handler, new VadProbabilityEventArgs(
                observation.Probability,
                observation.SmoothedProbability,
                observation.IsSpeech,
                observation.FrameIndex,
                timestamp,
                frameDuration));
        }

        private void HandleEvent(VadEvent evt)
        {
            switch (evt.Type)
            {
                case VadEventType.SpeechStart:
                {
                    long startF = Math.Max(1L, evt.StartFrame);
                    long endF = evt.EndFrame;

                    var frameAudio = ConcatFrameAudio(startF, endF);

                    // 取前置语音音频：仅取不与 frameAudio 重叠的部分
                    // (frameAudio 已包含 PadStartFrame 的回溯，使用绝对样本下标精确去重)
                    float[] preSpeechAudio = Array.Empty<float>();
                    if (_preSpeechBuffer != null && _preSpeechBuffer.Count > 0)
                    {
                        long bufferStartSampleIdx = _audioInputSampleCount - _preSpeechBuffer.Count;
                        long frameStartSampleIdx = (long)(startF - 1) * KaldiFbankExtractor.FrameShift;
                        long preSpeechCount = frameStartSampleIdx - bufferStartSampleIdx;
                        if (preSpeechCount > 0)
                        {
                            int take = (int)Math.Min(preSpeechCount, _preSpeechBuffer.Count);
                            preSpeechAudio = _preSpeechBuffer.DequeueRange(take);
                        }
                        _preSpeechBuffer.Clear();
                    }

                    float[] audio;
                    if (preSpeechAudio.Length == 0)
                    {
                        audio = frameAudio;
                    }
                    else
                    {
                        audio = new float[preSpeechAudio.Length + frameAudio.Length];
                        Buffer.BlockCopy(preSpeechAudio, 0, audio, 0, preSpeechAudio.Length * sizeof(float));
                        Buffer.BlockCopy(frameAudio, 0, audio, preSpeechAudio.Length * sizeof(float), frameAudio.Length * sizeof(float));
                    }

                    _speechBuffer.Clear();
                    _speechBuffer.AddRange(audio);
                    _stopwatch.Restart();
                    _isEmittingSpeech = true;
                    _lastEmittedFrame = endF;

                    SafeInvoke(OnSpeechStarted, EventArgs.Empty);
                    SafeInvoke(OnSpeaking, new VADEventArgs(audio, VADEventDataType.Chunk));
                    break;
                }
                case VadEventType.SpeechEnd:
                {
                    if (!_isEmittingSpeech)
                    {
                        // 异常情况下没有对应的 start，跳过
                        break;
                    }

                    long endF = evt.EndFrame;
                    if (_lastEmittedFrame < endF)
                    {
                        var tail = ConcatFrameAudio(_lastEmittedFrame + 1, endF);
                        if (tail.Length > 0)
                        {
                            _speechBuffer.AddRange(tail);
                            SafeInvoke(OnSpeaking, new VADEventArgs(tail, VADEventDataType.Chunk));
                        }
                        _lastEmittedFrame = endF;
                    }

                    _stopwatch.Stop();
                    SafeInvoke(OnSpeechEnded, new VADEventArgs(_speechBuffer.ToArray(), VADEventDataType.End, _stopwatch.Elapsed));

                    _isEmittingSpeech = false;
                    _speechBuffer.Clear();
                    _lastEmittedFrame = 0;
                    break;
                }
            }
        }

        private float[] ConcatFrameAudio(long startFrame, long endFrame)
        {
            if (startFrame > endFrame) return Array.Empty<float>();

            // 限定到环形缓冲当前可用范围。可用区间为
            // [_frameAudioRingWriteFrame - count, _frameAudioRingWriteFrame - 1]，其中 count 为已写入帧数。
            // writtenCount 受 capacity 上限约束 (≤ 容量), 转回 int 安全。
            int writtenCount = (int)Math.Min(_frameAudioRingWriteFrame - 1, _frameAudioRingCapacity);
            long oldestAvailable = _frameAudioRingWriteFrame - writtenCount;
            if (oldestAvailable < 1) oldestAvailable = 1;
            long newestAvailable = _frameAudioRingWriteFrame - 1;

            if (startFrame < oldestAvailable) startFrame = oldestAvailable;
            if (endFrame > newestAvailable) endFrame = newestAvailable;
            if (startFrame > endFrame) return Array.Empty<float>();

            const int frameShift = KaldiFbankExtractor.FrameShift;
            // 取数区间被夹到可用窗口内 (≤ capacity 帧), frameCount/startSlot 都落在 int 范围, 转回 int 安全。
            int frameCount = (int)(endFrame - startFrame + 1);
            int totalSamples = frameCount * frameShift;
            var result = new float[totalSamples];

            int startSlot = (int)((startFrame - 1) % _frameAudioRingCapacity);
            int slotsRemainingFromStart = _frameAudioRingCapacity - startSlot;

            if (frameCount <= slotsRemainingFromStart)
            {
                // 单段连续拷贝
                Buffer.BlockCopy(
                    _frameAudioRing,
                    startSlot * frameShift * sizeof(float),
                    result,
                    0,
                    totalSamples * sizeof(float));
            }
            else
            {
                // 跨边界，分两段拷贝
                int firstSamples = slotsRemainingFromStart * frameShift;
                Buffer.BlockCopy(
                    _frameAudioRing,
                    startSlot * frameShift * sizeof(float),
                    result,
                    0,
                    firstSamples * sizeof(float));

                int secondSamples = totalSamples - firstSamples;
                Buffer.BlockCopy(
                    _frameAudioRing,
                    0,
                    result,
                    firstSamples * sizeof(float),
                    secondSamples * sizeof(float));
            }

            return result;
        }

        /// <summary>
        /// 重置所有 VAD 状态（状态机、平滑窗口、帧/前置音频缓冲、fbank 剩余样本、当前发射段、ONNX 内部 caches）。
        /// 适用于切换音源、长时间静音后重建流式上下文等场景。
        /// 不会触发 <see cref="OnSpeechEnded"/> —— 若当前正在发射语音，调用方应自行处理收尾。
        /// </summary>
        public void Reset()
        {
            lock (_lock)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(FireRedVoiceActivityDetectService));

                // 状态机
                _state = VadState.Silence;
                _frameCnt = 0;
                _smoothWindow.Clear();
                _smoothSum = 0f;
                _probabilityFrameCnt = 0;
                _probabilitySmoothWindow.Clear();
                _probabilitySmoothSum = 0f;
                _speechCnt = 0;
                _silenceCnt = 0;
                _hitMaxSpeech = false;
                _lastSpeechStartFrame = -1;
                _lastSpeechEndFrame = -1;

                // 帧音频环形缓冲
                Array.Clear(_frameAudioRing, 0, _frameAudioRing.Length);
                _frameAudioRingWriteFrame = 1;

                // 前置语音缓冲与样本计数
                _preSpeechBuffer?.Clear();
                _audioInputSampleCount = 0;

                // fbank 跨调用样本剩余
                _fbankRemainder = Array.Empty<float>();

                // 发射状态
                _isEmittingSpeech = false;
                _speechBuffer.Clear();
                _lastEmittedFrame = 0;

                _stopwatch.Reset();

                // ONNX 流式 caches
                _onnx?.ResetCaches();
            }
        }

        private void SafeInvoke<T>(EventHandler<T>? handler, T args) where T : EventArgs
        {
            if (handler == null) return;
            try
            {
                handler.Invoke(this, args);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"事件处理异常: {ex.Message}");
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            lock (_lock)
            {
                if (_disposed) return;
                if (disposing)
                {
                    _onnx?.Dispose();
                    _onnx = null;
                }
                _disposed = true;
            }
        }
    }
}
