using System.Diagnostics;
using Microsoft.Extensions.Options;
using VoiceBox.APM.WebRtc.Internal;
using VoiceBox.APM.WebRtc.Models;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;

namespace VoiceBox.APM.WebRtc
{
    /// <summary>
    /// 基于 WebRTC M151 Audio Processing Module 的<b>统一采集侧处理器</b>。
    /// 单个 APM 实例在每次 <see cref="Process"/> 内依次完成
    /// 高通滤波(HPF) → 回声消除(AEC3) → 噪声抑制(NS) → 自动增益(AGC2)，
    /// 取代分离的 AEC + DNR，从而获得 AEC3 桌面级双讲处理能力。
    /// <para>
    /// 作为 <see cref="IAcousticEchoCancellationService"/> 挂到全双工音频回调中：用对齐的播放(远端)
    /// 信号作 AEC 参考。APM 按固定 10ms 块处理(16k=160样本)，与回调帧长通常不一致，故内部用 FIFO
    /// 重分块，并预填一个处理块的延迟以保证每次回调都能稳定取出等量已处理样本（净增 ~10ms 延迟）。
    /// </para>
    /// <para>当前仅支持单声道。<see cref="Process"/> 须在单一(实时)线程调用，不应与 <see cref="Reset"/>/
    /// <see cref="Dispose"/> 并发。<see cref="TryGetStats"/> 可跨线程调用(APM 内部加锁)。</para>
    /// </summary>
    public sealed class WebRtcApmService :
        IAcousticEchoCancellationService,
        IAcousticProcessingMetricsSource
    {
        private readonly uint _sampleRate;
        private readonly uint _channels;
        private readonly int _block;            // APM 每次处理的样本数 (10ms)
        private readonly ApmInterop.ApmConfig _nativeConfig;
        private readonly byte[]? _neuralResidualEchoModel;
        private readonly object _nativeLifecycleSync = new();

        private nint _handle;                   // 原生 APM 句柄(0 表示已释放)
        private readonly FloatFifo _renderFifo;
        private readonly FloatFifo _captureFifo;
        private readonly FloatFifo _cleanedFifo;

        // APM 去交织接口缓冲（单声道：[1][block]）。
        private readonly float[] _renderBlock;
        private readonly float[] _captureBlock;

        private bool _unsupportedLogged;
        private bool _errorLogged;
        private bool _disposed;

        public int DelaySamples => _block;

        static WebRtcApmService()
        {
            // 注册原生库解析器(从 runtimes/<rid>/native/ 加载 webrtc-apm)，须在任何 P/Invoke 之前。
            NativeLibraryLoader.EnsureRegistered();

            // 在任何实例创建前校验原生 ABI，陈旧/不匹配的原生库快速失败而非运行期崩溃。
            int native;
            try
            {
                native = ApmInterop.AbiVersion();
            }
            catch (DllNotFoundException ex)
            {
                throw new InvalidOperationException(
                    "Native library 'webrtc-apm' not found. Ensure the runtimes/<rid>/native binary is present " +
                    "(build it from native/webrtc-apm, see its README).", ex);
            }

            if (native != ApmInterop.ExpectedAbiVersion)
            {
                throw new InvalidOperationException(
                    $"webrtc-apm ABI mismatch: native={native}, expected={ApmInterop.ExpectedAbiVersion}. " +
                    "Rebuild the native shim from native/webrtc-apm.");
            }
        }

        public WebRtcApmService(IOptions<WebRtcApmOptions> options)
        {
            ArgumentNullException.ThrowIfNull(options);

            WebRtcApmOptions value = options.Value
                ?? throw new ArgumentException("WebRTC APM options value cannot be null.", nameof(options));
            value.Validate();

            _sampleRate = value.SampleRate;
            _channels = value.ChannelCount;
            _block = (int)(_sampleRate / 100); // 16k => 160
            _neuralResidualEchoModel = value.EnableNeuralResidualEchoEstimator
                ? NeuralResidualEchoModelLoader.Load(value.NeuralResidualEchoEstimatorModelPath)
                : null;

            _nativeConfig = new ApmInterop.ApmConfig
            {
                StructSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf<ApmInterop.ApmConfig>(),
                SampleRate = (int)_sampleRate,
                NumChannels = (int)_channels,
                EnableHighPassFilter = value.EnableHighPassFilter ? 1 : 0,
                EnableEchoCanceller = value.EnableEchoCanceller ? 1 : 0,
                EnableNoiseSuppression = value.EnableNoiseSuppression ? 1 : 0,
                NsLevel = (int)value.NoiseSuppressionLevel,
                EnableAgc2 = value.EnableAgc2 ? 1 : 0,
                Agc2AdaptiveDigital = value.Agc2AdaptiveDigital ? 1 : 0,
                // Delay is estimated from the live render/capture timelines. Do not inject a
                // hardware-specific startup delay into the AEC path.
                StreamDelayMs = 0,
                DtEnrThreshold = value.DominantNearendEnrThreshold,
                DtEnrExitThreshold = value.DominantNearendEnrExitThreshold,
                DtSnrThreshold = value.DominantNearendSnrThreshold,
                EpNearendLen = value.EchoPathNearendLen,
            };

            _handle = CreateNativeHandle("initial creation");

            int cap = Math.Max(16 * _block, 4096);
            _renderFifo = new FloatFifo(cap);
            _captureFifo = new FloatFifo(cap);
            _cleanedFifo = new FloatFifo(cap);
            _cleanedFifo.WriteZeros(_block); // 预填一个块的延迟，杜绝取数欠载

            _renderBlock = new float[_block];
            _captureBlock = new float[_block];
        }

        /// <inheritdoc />
        public void Process(Span<float> capture, ReadOnlySpan<float> render, uint channels, uint sampleRate)
        {
            if (_disposed || _handle == nint.Zero) return;

            // 当前仅支持单声道；其它情况原样返回(直通)。
            if (channels != 1 || _channels != 1)
            {
                if (!_unsupportedLogged)
                {
                    Debug.WriteLine("[WebRtcApmService] 仅支持单声道，已直通。");
                    _unsupportedLogged = true;
                }
                return;
            }

            // PortAudio Float32 已是 [-1,1]，与 APM 一致，直接送入。
            _renderFifo.Write(render);
            _captureFifo.Write(capture);

            while (_captureFifo.Count >= _block && _renderFifo.Count >= _block)
            {
                // 1) 远端参考 -> AEC（须先于 capture）。
                _renderFifo.Read(_renderBlock);
                // 2) 近端采集 -> 统一处理(HPF+AEC3+NS+AGC2，原地)。
                _captureFifo.Read(_captureBlock);

                if (!ProcessBlock())
                {
                    // 原生处理失败：保持本块直通(不写入 cleaned 的处理结果，改写回原采集块)，
                    // 避免实时线程抛异常或产出静音。
                    _cleanedFifo.Write(_captureBlock);
                    continue;
                }

                _cleanedFifo.Write(_captureBlock);
            }

            // 取出与本次回调等量的已处理样本，原地覆盖 capture。
            // 由于预填了一个块的延迟，cleanedFifo 始终有足够样本（数学上 prime=block ⇒ 不欠载）。
            if (_cleanedFifo.Count >= capture.Length)
            {
                _cleanedFifo.Read(capture);
            }
            // else: 仅可能在极端边界出现；此时保持 capture 原样(直通)，不阻塞实时线程。
        }

        /// <summary>对 <see cref="_renderBlock"/>/<see cref="_captureBlock"/> 跑一块 reverse+capture。成功返回 true。</summary>
        private unsafe bool ProcessBlock()
        {
            int rcRev, rcCap;
            fixed (float* render = _renderBlock)
            fixed (float* capture = _captureBlock)
            {
                float* renderCh = render;
                float* captureCh = capture;
                rcRev = ApmInterop.ProcessReverse(_handle, &renderCh, 1, _block);
                rcCap = ApmInterop.ProcessCapture(_handle, &captureCh, 1, _block);
            }

            if (rcRev != ApmInterop.Ok || rcCap != ApmInterop.Ok)
            {
                if (!_errorLogged)
                {
                    Debug.WriteLine($"[WebRtcApmService] 原生处理返回非零 (reverse={rcRev}, capture={rcCap})，本块直通。");
                    _errorLogged = true;
                }
                return false;
            }
            return true;
        }

        /// <summary>
        /// 读取最近一次的 APM 统计(ERLE/延迟等)。可跨线程调用。
        /// 成功返回 true 并填充 <paramref name="stats"/>。
        /// </summary>
        public bool TryGetStats(out ApmStatistics stats)
        {
            lock (_nativeLifecycleSync)
            {
                if (_disposed || _handle == nint.Zero)
                {
                    stats = default;
                    return false;
                }

                if (ApmInterop.GetStats(_handle, out ApmInterop.ApmStats s) != ApmInterop.Ok)
                {
                    stats = default;
                    return false;
                }

                stats = new ApmStatistics(
                    HasEchoReturnLoss: s.HasEchoReturnLoss != 0,
                    EchoReturnLoss: s.EchoReturnLoss,
                    HasEchoReturnLossEnhancement: s.HasEchoReturnLossEnhancement != 0,
                    EchoReturnLossEnhancement: s.EchoReturnLossEnhancement,
                    HasDelayMs: s.HasDelayMs != 0,
                    DelayMs: s.DelayMs,
                    HasDelayMedianMs: s.HasDelayMedianMs != 0,
                    DelayMedianMs: s.DelayMedianMs,
                    HasResidualEchoLikelihood: s.HasResidualEchoLikelihood != 0,
                    ResidualEchoLikelihood: s.ResidualEchoLikelihood,
                    HasResidualEchoLikelihoodRecentMax: s.HasResidualEchoLikelihoodRecentMax != 0,
                    ResidualEchoLikelihoodRecentMax: s.ResidualEchoLikelihoodRecentMax,
                    HasDivergentFilterFraction: s.HasDivergentFilterFraction != 0,
                    DivergentFilterFraction: s.DivergentFilterFraction);
                return true;
            }
        }

        public bool TryGetLatestMetrics(out AcousticProcessingMetrics metrics)
        {
            if (!TryGetStats(out ApmStatistics stats))
            {
                metrics = default;
                return false;
            }

            metrics = new AcousticProcessingMetrics(
                stats.HasEchoReturnLoss,
                stats.EchoReturnLoss,
                stats.HasEchoReturnLossEnhancement,
                stats.EchoReturnLossEnhancement,
                stats.HasDelayMs,
                stats.DelayMs,
                stats.HasDelayMedianMs,
                stats.DelayMedianMs,
                stats.HasResidualEchoLikelihood,
                stats.ResidualEchoLikelihood,
                Stopwatch.GetTimestamp());
            return true;
        }

        /// <inheritdoc />
        public void Reset()
        {
            lock (_nativeLifecycleSync)
            {
                if (_disposed || _handle == nint.Zero) return;

                _renderFifo.Clear();
                _captureFifo.Clear();
                _cleanedFifo.Clear();
                _cleanedFifo.WriteZeros(_block);

                nint replacement = CreateNativeHandle("stream reset");

                nint previous = _handle;
                _handle = replacement;
                ApmInterop.Destroy(previous);
                _unsupportedLogged = false;
                _errorLogged = false;
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        // 终结器兜底：即便调用方漏调 Dispose，也确保原生句柄被释放，杜绝原生内存泄漏。
        ~WebRtcApmService() => Dispose(false);

        private unsafe nint CreateNativeHandle(string operation)
        {
            nint handle;
            int rc;
            if (_neuralResidualEchoModel is null)
            {
                rc = ApmInterop.Create(in _nativeConfig, out handle);
            }
            else
            {
                fixed (byte* model = _neuralResidualEchoModel)
                {
                    rc = ApmInterop.CreateWithNeuralResidualEchoModel(
                        in _nativeConfig,
                        model,
                        (ulong)_neuralResidualEchoModel.Length,
                        out handle);
                }
            }

            if (rc == ApmInterop.Ok && handle != nint.Zero)
                return handle;

            if (handle != nint.Zero)
                ApmInterop.Destroy(handle);

            string detail = rc == ApmInterop.ErrModel
                ? "The TFLite model is incompatible with this libwebrtc build or failed estimator validation."
                : "See native status codes in apm_shim.h.";
            throw new InvalidOperationException(
                $"Failed to create WebRTC APM during {operation} (rc={rc}). {detail}");
        }

        private void Dispose(bool disposing)
        {
            lock (_nativeLifecycleSync)
            {
                if (_disposed) return;
                _disposed = true;

                nint handle = System.Threading.Interlocked.Exchange(ref _handle, nint.Zero);
                if (handle != nint.Zero)
                {
                    ApmInterop.Destroy(handle);
                }
            }
        }
    }
}
