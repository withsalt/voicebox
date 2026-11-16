using Microsoft.Extensions.Options;
using VoiceBox.AEC.DFSMN;
using VoiceBox.AEC.DFSMN.Models;
using VoiceBox.APM.WebRtc;
using VoiceBox.APM.WebRtc.Models;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.DNR.DeepFilterNet;
using VoiceBox.DNR.DeepFilterNet.Models;
using VoiceBox.DNR.DPDFNet;
using VoiceBox.DNR.DPDFNet.Models;
using VoiceBox.DNR.Hush;
using VoiceBox.DNR.Hush.Models;
using VoiceBox.DNR.RNNoise;

namespace VoiceBox.ASR.Qwen3.Benchmark
{
    internal enum PreprocMode { None, Aec, Dnr, Both }

    /// <summary>DNR 后端实现。DPDFNet=DeepFilterNet 系 ONNX; DeepFilterNet=DFN3-LL ONNX 流式 (48k); Hush=DFN3-SE ONNX 流式 (16k); RNNoise=经典 GRU 原生库 (48k); WebRtc=WebRTC APM 噪声抑制 (原生)。</summary>
    internal enum DnrBackend { DpdfNet, RNNoise, WebRtc, DeepFilterNet, Hush }

    /// <summary>
    /// 前处理旋钮。默认值对齐 VoiceAssistantDemo: DFSMN AEC (MaskFloorDb=60) +
    /// DPDFNet DpDfNet8 (AttenuationLimitDb=6)。EP 默认 CPU, 避免与 Qwen3-ASR 抢 GPU。
    /// </summary>
    internal sealed class PreprocParams
    {
        public OnnxExecutionProvider ExecutionProvider { get; init; } = OnnxExecutionProvider.Cpu;

        /// <summary>AEC 神经掩码抑制下限 (dB)。null = 不设限 (允许模型掩码原样, 最激进)。Demo 默认 60。</summary>
        public float? MaskFloorDb { get; init; } = 60f;

        /// <summary>DNR 后端: DPDFNet (默认, 对齐 Demo) 或 RNNoise。</summary>
        public DnrBackend DnrBackend { get; init; } = DnrBackend.DpdfNet;

        /// <summary>DNR 模型档位 (仅 DPDFNet 后端生效)。Demo 默认 DpDfNet8 (最强)。</summary>
        public DPDFNetModel DnrModel { get; init; } = DPDFNetModel.DpDfNet8;

        /// <summary>DNR 单帧最大抑制量上限 (dB, 仅 DPDFNet 后端生效)。null = 不设限。Demo 默认 6。</summary>
        public float? AttenuationLimitDb { get; init; } = 6f;

        /// <summary>AttenuationLimitDb 的施加方式 (仅 DPDFNet 后端生效)。默认复数混合 (历史行为)。</summary>
        public DPDFNetAttenuationMode AttenuationMode { get; init; } = DPDFNetAttenuationMode.ComplexBlend;

    }

    /// <summary>
    /// 把 AEC / DNR 以"离线整段"的方式套到一条 utterance 上, 用于 benchmark 的前处理矩阵。
    ///
    /// 关键点 (与实时全双工链路的差异):
    /// <list type="bullet">
    ///   <item>AEC 的远端(render)参考<b>恒为全零</b> —— benchmark 无 TTS 回放。这正好复刻
    ///   "用户对着安静的助手说话、无回声"的最常见场景, 也是判断 AEC 神经掩码是否在干净语音上
    ///   掉点的关键工况。</item>
    ///   <item>两者都有 ~一帧的流水线延迟与需要冲洗的尾部; 这里整段喂入后再用零做 flush,
    ///   把尾部样本取回, 避免截断 utterance 末尾 (可能含句尾辅音)。</item>
    ///   <item>每条 utterance 前复位状态, 条与条之间互不串扰。</item>
    /// </list>
    /// 代价是首尾各多出约 40ms 静音, 对离线 ASR 无害 (RTF 仍按原始音频时长计算)。
    /// </summary>
    internal sealed class AudioPreprocessor : IDisposable
    {
        private const uint SampleRate = 16000;
        private const int AecLatencySamples = DfsmnAecOptions.DefaultFeatureFrameLength; // 640, 预置的前导静音
        private const int AecFlushSamples = 1280;  // > frameLength, 确保流水线尾部排空
        private const int ApmBlockSamples = 160;   // WebRTC APM 10ms 块; cleanedFifo 预填一块零 ⇒ 1 块延迟
        private const int ApmFlushSamples = 480;   // > 1 块, 排空 FIFO 内滞后的真实尾部

        private readonly PreprocMode _mode;
        private readonly DfsmnAecService? _aec;
        private readonly IDigitalNoiseReductionService? _dnr; // DPDFNet / RNNoise / DeepFilterNet / Hush, 同一 Process 契约
        private readonly WebRtcApmService? _apm;   // DnrBackend.WebRtc: 统一 APM (HPF+AEC3+NS), 对齐 VoiceAssistantDemo 生产链路

        public PreprocMode Mode => _mode;

        public AudioPreprocessor(PreprocMode mode, PreprocParams p)
        {
            ArgumentNullException.ThrowIfNull(p);
            _mode = mode;

            bool needAec = mode is PreprocMode.Aec or PreprocMode.Both;
            bool needDnr = mode is PreprocMode.Dnr or PreprocMode.Both;

            if (needAec)
            {
                _aec = new DfsmnAecService(Options.Create(new DfsmnAecOptions
                {
                    SampleRate = SampleRate,
                    ChannelCount = 1,
                    ExecutionProvider = p.ExecutionProvider,
                    MaskFloorDb = p.MaskFloorDb,
                }));
            }

            if (needDnr)
            {
                if (p.DnrBackend == DnrBackend.WebRtc)
                {
                    // 与 VoiceAssistantDemo 一致: HPF + AEC3 + NS(High), AGC2 关闭。
                    // benchmark 无 TTS 回放, render 恒零 ⇒ AEC3 近似透明但仍在跑增益估计/延迟补偿,
                    // 实测的是整条 APM 链路 (主要为 HPF+NS) 对干净语音的损伤; 只想测 HPF+NS 可关 EC。
                    _apm = new WebRtcApmService(Options.Create(new WebRtcApmOptions
                    {
                        SampleRate = SampleRate,
                        ChannelCount = 1,
                        EnableHighPassFilter = true,
                        EnableEchoCanceller = true,
                        EnableNoiseSuppression = true,
                        EnableAgc2 = false,
                    }));
                }
                else
                {
                    _dnr = p.DnrBackend switch
                    {
                        DnrBackend.RNNoise => new RNNoiseDNRService(Options.Create(new RNNoiseOptions
                        {
                            SampleRate = SampleRate,   // 16k 输入, 服务内部重采样到 48k 跑 RNNoise 再回 16k
                            ChannelCount = 1,
                        })),
                        DnrBackend.DeepFilterNet => new DeepFilterNetDNRService(Options.Create(new DeepFilterNetOptions
                        {
                            SampleRate = SampleRate,   // 16k 输入, 服务内部重采样到 48k 原生率再回 16k
                            ChannelCount = 1,
                            ExecutionProvider = p.ExecutionProvider,
                        })),
                        DnrBackend.Hush => new HushDNRService(Options.Create(new HushOptions
                        {
                            SampleRate = SampleRate,   // Hush 原生 16k, 零重采样
                            ChannelCount = 1,
                            ExecutionProvider = p.ExecutionProvider,
                        })),
                        _ => new DPDFNetDNRService(Options.Create(new DPDFNetOptions
                        {
                            SampleRate = SampleRate,
                            ChannelCount = 1,
                            ExecutionProvider = p.ExecutionProvider,
                            Model = p.DnrModel,
                            AttenuationLimitDb = p.AttenuationLimitDb,
                            AttenuationMode = p.AttenuationMode,
                        })),
                    };
                }
            }
        }

        /// <summary>
        /// 处理一条 utterance, 返回前处理后的新数组 (16 kHz mono f32)。None 模式直接返回输入副本。
        /// 链路顺序与实时端一致: Mic → AEC → DNR。
        /// </summary>
        public float[] Process(float[] input)
        {
            ArgumentNullException.ThrowIfNull(input);
            if (_mode == PreprocMode.None || input.Length == 0)
                return input;

            float[] audio = input;
            if (_aec is not null) audio = ApplyAec(audio);
            if (_dnr is not null) audio = ApplyDnr(audio);
            if (_apm is not null) audio = ApplyApm(audio);
            return audio;
        }

        /// <summary>
        /// AEC: render 恒零, 整段 in-place 处理后再 flush。输出 FIFO 在 Reset 时预置了
        /// <see cref="AecLatencySamples"/> 个零, 它们必然最先被读出, 故可安全裁掉这段前导静音对齐起点。
        /// </summary>
        private float[] ApplyAec(float[] input)
        {
            _aec!.Reset();

            int n = input.Length;
            float[] capture = (float[])input.Clone();
            float[] render = new float[n]; // 全零远端参考
            _aec.Process(capture, render, 1, SampleRate);

            // 用零继续驱动, 把流水线里滞后的真实尾部排空。
            float[] flush = new float[AecFlushSamples];
            float[] flushRender = new float[AecFlushSamples];
            _aec.Process(flush, flushRender, 1, SampleRate);

            float[] acc = new float[n + AecFlushSamples];
            Array.Copy(capture, 0, acc, 0, n);
            Array.Copy(flush, 0, acc, n, AecFlushSamples);

            // 裁掉可证明为零的前导静音 (输出 FIFO 预置), 让起点对齐; utterance 远长于 640 故恒成立。
            return acc.Length > AecLatencySamples ? acc[AecLatencySamples..] : acc;
        }

        private float[] ApplyDnr(float[] input)
            => _dnr!.ProcessAll(input);

        /// <summary>
        /// WebRTC APM: render 恒零, in-place 处理后用零尾排空 FIFO 中滞后的真实尾部。
        /// cleanedFifo 预填一个 10ms 块 ⇒ 输出整体滞后 160 样本, 裁掉该前导零对齐起点。
        /// </summary>
        private float[] ApplyApm(float[] input)
        {
            _apm!.Reset();

            int n = input.Length;
            float[] capture = (float[])input.Clone();
            float[] render = new float[n]; // 全零远端参考
            _apm.Process(capture, render, 1, SampleRate);

            float[] flush = new float[ApmFlushSamples];
            float[] flushRender = new float[ApmFlushSamples];
            _apm.Process(flush, flushRender, 1, SampleRate);

            float[] acc = new float[n + ApmFlushSamples];
            Array.Copy(capture, 0, acc, 0, n);
            Array.Copy(flush, 0, acc, n, ApmFlushSamples);

            return acc.Length > ApmBlockSamples ? acc[ApmBlockSamples..] : acc;
        }

        public void Dispose()
        {
            _aec?.Dispose();
            _dnr?.Dispose();
            _apm?.Dispose();
        }

        public static PreprocMode ParseMode(string s) => s.Trim().ToLowerInvariant() switch
        {
            "none" => PreprocMode.None,
            "aec" => PreprocMode.Aec,
            "dnr" => PreprocMode.Dnr,
            "both" => PreprocMode.Both,
            _ => throw new ArgumentException($"Unknown preproc mode: {s} (none|aec|dnr|both)"),
        };

        public static string ModeLabel(PreprocMode m) => m switch
        {
            PreprocMode.None => "none",
            PreprocMode.Aec => "aec",
            PreprocMode.Dnr => "dnr",
            PreprocMode.Both => "both",
            _ => m.ToString().ToLowerInvariant(),
        };

        public static DnrBackend ParseBackend(string s) => s.Trim().ToLowerInvariant() switch
        {
            "dpdfnet" => DnrBackend.DpdfNet,
            "deepfilternet" or "dfn" => DnrBackend.DeepFilterNet,
            "hush" => DnrBackend.Hush,
            "rnnoise" => DnrBackend.RNNoise,
            "webrtc" => DnrBackend.WebRtc,
            _ => throw new ArgumentException($"Unknown DNR backend: {s} (dpdfnet|deepfilternet|hush|rnnoise|webrtc)"),
        };

        public static string BackendLabel(DnrBackend b) => b switch
        {
            DnrBackend.RNNoise => "rnnoise",
            DnrBackend.WebRtc => "webrtc",
            DnrBackend.DeepFilterNet => "deepfilternet",
            DnrBackend.Hush => "hush",
            _ => "dpdfnet",
        };
    }
}
