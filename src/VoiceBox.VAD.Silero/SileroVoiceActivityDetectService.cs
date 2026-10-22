using System.Diagnostics;
using Microsoft.Extensions.Options;
using VoiceBox.Core.Models;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.VadService.Models;
using VoiceBox.VAD.Silero.Models;

namespace VoiceBox.VAD.Silero
{
    public sealed class SileroVoiceActivityDetectService : IVoiceActivityDetectService
    {
        public event EventHandler<EventArgs>? OnSpeechStarted;
        public event EventHandler<VADEventArgs>? OnSpeaking;
        public event EventHandler<VADEventArgs>? OnSpeechEnded;
        public event EventHandler<VadProbabilityEventArgs>? OnProbability;

        private readonly SileroVadOptions _vadOptions;
        private readonly int _minSilenceFrames;
        private readonly int _windowSizeSample;
        // 单段语音最大累积样本数 (0 = 不限制)。防止持续被判为语音时 _speechBuffer 无限增长导致 OOM。
        private readonly int _maxSpeechSamples;

        private readonly RingBufferQueue<float> _preSpeechBuffer;
        private readonly RingBufferQueue<float> _bufferQueue;

        private bool _isSpeaking = false;
        private int _silenceFrameCount = 0;
        private long _frameCount;
        private readonly List<float> _speechBuffer;

        private SileroVadOnnxRunner? _vadOnnxRunner;
        private bool _disposed = false;

        private readonly object _lock = new object();
        private readonly Stopwatch _stopwatch = new Stopwatch();

        public SileroVoiceActivityDetectService(IOptions<SileroVadOptions> options)
        {
            _vadOptions = options?.Value ?? throw new ArgumentNullException(nameof(options));

            // 窗口尺寸 (_windowSizeSample) 与推理器期望只在 8k/16k 两个原生采样率下吻合: 16k 取 512,
            // 其余取 256 (为 8k 设计)。对 16k 的整数倍 (如 48k), 推理器 ValidateInput 会先按 step
            // 降采样到 16k 再要求 512 样本, 而服务给出 256, 首帧即抛 ArgumentException。这里前置白名单
            // 校验, 把深层崩溃转为构造期清晰报错; 需要其它采样率请在上游重采样到 16k。
            if (_vadOptions.SampleRate != 8000 && _vadOptions.SampleRate != 16000)
                throw new NotSupportedException(
                    $"Silero VAD 仅支持 SampleRate 8000 或 16000 (实际 {_vadOptions.SampleRate})。请在 upstream 将音频重采样到 16k。");

            _windowSizeSample = _vadOptions.SampleRate == 16000 ? 512 : 256;

            int preSpeechBufferSize = (int)(_vadOptions.SampleRate * (_vadOptions.PreSpeechPaddingMs / 1000f));
            _preSpeechBuffer = new RingBufferQueue<float>(preSpeechBufferSize);

            _bufferQueue = new RingBufferQueue<float>((int)_vadOptions.SampleRate * 2);
            _speechBuffer = new List<float>((int)_vadOptions.SampleRate * 10);

            float windowDurationMs = (_windowSizeSample / (float)_vadOptions.SampleRate) * 1000f;
            _minSilenceFrames = (int)Math.Ceiling(_vadOptions.MinSilenceDurationMs / windowDurationMs);

            _maxSpeechSamples = _vadOptions.MaxSpeechDurationMs > 0
                ? (int)((long)_vadOptions.SampleRate * _vadOptions.MaxSpeechDurationMs / 1000)
                : 0;

            if (string.IsNullOrWhiteSpace(_vadOptions.VADModelPath))
                throw new ArgumentNullException(nameof(_vadOptions.VADModelPath), "VADModelPath cannot be null or empty.");
            if (!File.Exists(_vadOptions.VADModelPath))
                throw new FileNotFoundException($"VAD Model file not found: {_vadOptions.VADModelPath}");
            _vadOnnxRunner = new SileroVadOnnxRunner(_vadOptions.VADModelPath, _vadOptions);
        }

        public void Predict(ReadOnlySpan<float> audioData)
        {
            lock (_lock)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(SileroVoiceActivityDetectService));
                if (audioData.IsEmpty)
                    return;

                bool hasSpeechSubscribers = OnSpeaking != null || OnSpeechStarted != null || OnSpeechEnded != null;
                bool hasProbabilitySubscribers = OnProbability != null;
                if (!hasSpeechSubscribers && !hasProbabilitySubscribers)
                    return;

                _bufferQueue.EnqueueRange(audioData);

                while (_bufferQueue.Count >= _windowSizeSample)
                {
                    float[] audioChunk = _bufferQueue.DequeueRange(_windowSizeSample);
                    ProcessVadResult(audioChunk, hasSpeechSubscribers);
                }
            }
        }

        private void ProcessVadResult(float[] audioChunk, bool hasSpeechSubscribers)
        {
            if (_vadOnnxRunner == null)
            {
                throw new InvalidOperationException("VAD runner is not initialized.");
            }

            float speechProbability = _vadOnnxRunner.Call(new[] { audioChunk }, (int)_vadOptions.SampleRate)[0];
            if (!float.IsFinite(speechProbability))
                speechProbability = 0f;
            speechProbability = Math.Clamp(speechProbability, 0f, 1f);

            _frameCount++;
            bool isSpeech = speechProbability >= this._vadOptions.Threshold;
            RaiseProbability(speechProbability, isSpeech);

            if (!hasSpeechSubscribers)
                return;

            if (isSpeech)
            {
                _silenceFrameCount = 0;

                if (!_isSpeaking)
                {
                    _isSpeaking = true;
                    _stopwatch.Restart();

                    SafeInvoke(OnSpeechStarted, EventArgs.Empty);

                    if (_preSpeechBuffer.Count > 0)
                    {
                        var preSpeechData = _preSpeechBuffer.ToArray();
                        _speechBuffer.AddRange(preSpeechData);
                        _preSpeechBuffer.Clear();
                    }

                    _speechBuffer.AddRange(audioChunk);

                    var startingChunk = new float[_speechBuffer.Count];
                    _speechBuffer.CopyTo(startingChunk);

                    SafeInvoke(OnSpeaking, new VADEventArgs(startingChunk, VADEventDataType.Chunk));
                }
                else
                {
                    _speechBuffer.AddRange(audioChunk);
                    SafeInvoke(OnSpeaking, new VADEventArgs(audioChunk, VADEventDataType.Chunk));
                }

                // 强制结束超长语音段: 当 VAD 持续把输入判为语音时, 上面的分支会每帧向 _speechBuffer
                // 追加样本而永不触发静音结束路径 (line ~134 的 Clear), 导致缓冲无限增长直至 OOM。
                // 到达上限即结束当前段 (与 FireRedVAD 的 MaxSpeechFrame 一致), 下一帧语音会重新开始新段。
                if (_maxSpeechSamples > 0 && _isSpeaking && _speechBuffer.Count >= _maxSpeechSamples)
                {
                    _stopwatch.Stop();
                    SafeInvoke(OnSpeechEnded, new VADEventArgs(_speechBuffer.ToArray(), VADEventDataType.End, _stopwatch.Elapsed));

                    _isSpeaking = false;
                    _silenceFrameCount = 0;
                    _speechBuffer.Clear();
                    _preSpeechBuffer.Clear();
                }
            }
            else
            {
                // 未检测到语音 (静音)
                if (_isSpeaking)
                {
                    _silenceFrameCount++;
                    // 将尾部静音也暂时录进去
                    _speechBuffer.AddRange(audioChunk);

                    if (_silenceFrameCount >= _minSilenceFrames)
                    {
                        _stopwatch.Stop();

                        SafeInvoke(OnSpeechEnded, new VADEventArgs(_speechBuffer.ToArray(), VADEventDataType.End, _stopwatch.Elapsed));

                        // 重置状态
                        _isSpeaking = false;
                        _silenceFrameCount = 0;
                        _speechBuffer.Clear();
                        _preSpeechBuffer.Clear();
                    }
                }
                else
                {
                    _preSpeechBuffer.EnqueueRange(audioChunk);
                }
            }
        }

        private void SafeInvoke<T>(EventHandler<T>? handler, T args) where T : EventArgs
        {
            if (handler == null) return;
            try
            {
                handler?.Invoke(this, args);
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
                    // 释放托管资源
                    _vadOnnxRunner?.Dispose();
                    _vadOnnxRunner = null;
                }
                _disposed = true;
            }
        }

        public void Reset()
        {
            lock (_lock)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(SileroVoiceActivityDetectService));

                // 状态机与缓冲: 复位发射状态、静音计数、语音/前置/分块缓冲。
                _isSpeaking = false;
                _silenceFrameCount = 0;
                _frameCount = 0;
                _speechBuffer.Clear();
                _preSpeechBuffer.Clear();
                _bufferQueue.Clear();
                _stopwatch.Reset();

                // ONNX 循环隐藏状态 (LSTM): 不清会把上一段上下文带到新流, 造成误检。
                _vadOnnxRunner?.ResetStates();
            }
        }

        private void RaiseProbability(float probability, bool isSpeech)
        {
            EventHandler<VadProbabilityEventArgs>? handler = OnProbability;
            if (handler == null)
                return;

            double secondsPerFrame = _windowSizeSample / (double)_vadOptions.SampleRate;
            var timestamp = TimeSpan.FromSeconds((_frameCount - 1) * secondsPerFrame);
            var frameDuration = TimeSpan.FromSeconds(secondsPerFrame);
            SafeInvoke(handler, new VadProbabilityEventArgs(
                probability,
                probability,
                isSpeech,
                _frameCount,
                timestamp,
                frameDuration));
        }
    }
}
