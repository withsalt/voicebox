using System.Diagnostics;
using Microsoft.Extensions.Options;
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

namespace VoiceBox.DNR.Benchmark
{
    internal enum DnrKind { DeepFilterNet, DpdfNet, Hush, RNNoise, WebRtc }

    /// <summary>一个待评测后端的完整配置。Tier 仅 DpdfNet 用; NsLevel 仅 WebRtc 用。</summary>
    internal sealed record DnrSpec(DnrKind Kind, string Label, DPDFNetModel Tier, NoiseSuppressionLevel NsLevel);

    /// <summary>
    /// 统一的 DNR 引擎抽象, 把两种不同的运行契约收敛到一个 <see cref="Run"/>:
    ///   - 流式服务 (<see cref="IDigitalNoiseReductionService"/>): 通过 <c>ProcessAll</c> 逐帧处理并排尾;
    ///   - WebRTC APM: 逐 10ms 块 <c>Process(capture, zeroRender)</c>, 末尾零尾排空。
    /// <see cref="Run"/> 在 <paramref name="perFrameMs"/> 记录每个实时帧/块的处理耗时 (不含 flush),
    /// 供延迟分位统计。
    /// </summary>
    internal abstract class DnrEngine : IDisposable
    {
        public abstract int FrameSize { get; }
        public abstract string Label { get; }

        /// <summary>整段处理一条 utterance, 返回 enhanced (新数组)。逐帧耗时写入 perFrameMs (若非空)。</summary>
        public abstract float[] Run(float[] input, List<double>? perFrameMs);

        /// <summary>条与条之间复位状态, 互不串扰。</summary>
        public abstract void Reset();

        public abstract void Dispose();
    }

    /// <summary>包 <see cref="IDigitalNoiseReductionService"/> (DeepFilterNet / DPDFNet / Hush / RNNoise)。</summary>
    internal sealed class StreamingDnrEngine : DnrEngine
    {
        private readonly IDigitalNoiseReductionService _svc;
        public override string Label { get; }
        public override int FrameSize => Math.Max(1, _svc.FrameSize);

        public StreamingDnrEngine(IDigitalNoiseReductionService svc, string label)
        {
            _svc = svc;
            Label = label;
        }

        public override float[] Run(float[] input, List<double>? perFrameMs)
        {
            int fs = FrameSize;
            var outBuf = new float[input.Length];
            var work = new float[fs];

            _svc.Reset();
            int pos = 0;
            int outputCount = 0;
            var sw = new Stopwatch();
            try
            {
                while (pos < input.Length)
                {
                    int len = Math.Min(fs, input.Length - pos);
                    Array.Copy(input, pos, work, 0, len);
                    sw.Restart();
                    int produced = _svc.Process(work.AsSpan(0, len));
                    sw.Stop();
                    perFrameMs?.Add(sw.Elapsed.TotalMilliseconds);
                    if (produced > outBuf.Length - outputCount)
                        throw new InvalidOperationException("DNR produced more source-backed output than input.");
                    work.AsSpan(0, produced).CopyTo(outBuf.AsSpan(outputCount));
                    outputCount += produced;
                    pos += len;
                }

                long drainBudget = Math.Max(0, _svc.DelaySamples) + 4L * fs;
                long drainedInput = 0;
                while (outputCount < outBuf.Length)
                {
                    work.AsSpan().Clear();
                    int produced = _svc.Process(work);
                    int copyLength = Math.Min(produced, outBuf.Length - outputCount);
                    work.AsSpan(0, copyLength).CopyTo(outBuf.AsSpan(outputCount));
                    outputCount += copyLength;
                    drainedInput += work.Length;
                    if (drainedInput > drainBudget && outputCount < outBuf.Length)
                    {
                        throw new InvalidOperationException(
                            "DNR did not release its source-backed tail within the declared delay bound.");
                    }
                }

                return outBuf;
            }
            finally
            {
                _svc.Reset();
            }
        }

        public override void Reset() => _svc.Reset();

        public override void Dispose() => _svc.Dispose();
    }

    /// <summary>包 <see cref="WebRtcApmService"/> 作纯噪声抑制器 (EC 关 / NS 开 / HPF 开 / AGC 关, render 恒零)。</summary>
    internal sealed class WebRtcNsEngine : DnrEngine
    {
        private const uint SampleRate = 16000;
        private const int BlockSamples = 160; // 10ms @16k
        private const int FlushSamples = 480; // 排空 cleanedFifo 内滞后的真实尾部

        private readonly WebRtcApmService _apm;
        public override string Label { get; }
        public override int FrameSize => BlockSamples;

        public WebRtcNsEngine(WebRtcApmService apm, string label)
        {
            _apm = apm;
            Label = label;
        }

        public override float[] Run(float[] input, List<double>? perFrameMs)
        {
            var outBuf = new List<float>(input.Length + FlushSamples);
            var work = new float[BlockSamples];
            var zero = new float[BlockSamples];

            int pos = 0;
            var sw = new Stopwatch();
            while (pos < input.Length)
            {
                int len = Math.Min(BlockSamples, input.Length - pos);
                Array.Copy(input, pos, work, 0, len);
                sw.Restart();
                _apm.Process(work.AsSpan(0, len), zero.AsSpan(0, len), 1, SampleRate);
                sw.Stop();
                perFrameMs?.Add(sw.Elapsed.TotalMilliseconds);
                for (int i = 0; i < len; i++) outBuf.Add(work[i]);
                pos += len;
            }

            var flush = new float[FlushSamples];
            var flushRender = new float[FlushSamples];
            _apm.Process(flush.AsSpan(), flushRender.AsSpan(), 1, SampleRate);
            outBuf.AddRange(flush);

            return outBuf.ToArray();
        }

        public override void Reset() => _apm.Reset();

        public override void Dispose() => _apm.Dispose();
    }

    internal static class DnrEngineFactory
    {
        /// <summary>按 spec 构造引擎。ONNX 后端用 <paramref name="ep"/>; RNNoise/WebRtc 为 native, 忽略 EP。</summary>
        public static DnrEngine Create(DnrSpec spec, uint sampleRate, OnnxExecutionProvider ep)
        {
            switch (spec.Kind)
            {
                case DnrKind.DeepFilterNet:
                    return new StreamingDnrEngine(new DeepFilterNetDNRService(Options.Create(new DeepFilterNetOptions
                    {
                        SampleRate = sampleRate,
                        ChannelCount = 1,
                        ExecutionProvider = ep,
                    })), spec.Label);

                case DnrKind.DpdfNet:
                    return new StreamingDnrEngine(new DPDFNetDNRService(Options.Create(new DPDFNetOptions
                    {
                        SampleRate = sampleRate,
                        ChannelCount = 1,
                        ExecutionProvider = ep,
                        Model = spec.Tier,
                        // AttenuationLimitDb 默认 null（不设上限）= 完全采用增强谱。
                    })), spec.Label);

                case DnrKind.Hush:
                    return new StreamingDnrEngine(new HushDNRService(Options.Create(new HushOptions
                    {
                        SampleRate = sampleRate,
                        ChannelCount = 1,
                        ExecutionProvider = ep,
                    })), spec.Label);

                case DnrKind.RNNoise:
                    return new StreamingDnrEngine(new RNNoiseDNRService(Options.Create(new RNNoiseOptions
                    {
                        SampleRate = sampleRate,   // 库内部重采样到 48k 跑 RNNoise 再回该率
                        ChannelCount = 1,
                    })), spec.Label);

                case DnrKind.WebRtc:
                    return new WebRtcNsEngine(new WebRtcApmService(Options.Create(new WebRtcApmOptions
                    {
                        SampleRate = sampleRate,   // 须为 8/16/32/48k; benchmark 固定 16k
                        ChannelCount = 1,
                        EnableHighPassFilter = true,
                        EnableEchoCanceller = false,       // 无回声场景, 关 EC = 纯 NS+HPF
                        EnableNoiseSuppression = true,
                        NoiseSuppressionLevel = spec.NsLevel,
                        EnableAgc2 = false,
                    })), spec.Label);

                default:
                    throw new ArgumentOutOfRangeException(nameof(spec), spec.Kind, "Unknown DNR kind.");
            }
        }
    }

    /// <summary>解析 <c>--backends</c> token 列表为 <see cref="DnrSpec"/> 列表。</summary>
    internal static class BackendParser
    {
        public static readonly string[] DefaultTokens =
            { "deepfilternet", "dpdfnet", "hush", "rnnoise", "webrtc" };

        public static List<DnrSpec> Parse(IEnumerable<string> tokens, NoiseSuppressionLevel nsLevel)
        {
            var specs = new List<DnrSpec>();
            foreach (var raw in tokens)
            {
                string t = raw.Trim().ToLowerInvariant();
                if (t.Length == 0) continue;
                switch (t)
                {
                    case "deepfilternet":
                    case "dfn":
                        specs.Add(new DnrSpec(DnrKind.DeepFilterNet, "deepfilternet", DPDFNetModel.DpDfNet8, nsLevel));
                        break;
                    case "hush":
                        specs.Add(new DnrSpec(DnrKind.Hush, "hush", DPDFNetModel.DpDfNet8, nsLevel));
                        break;
                    case "rnnoise":
                        specs.Add(new DnrSpec(DnrKind.RNNoise, "rnnoise", DPDFNetModel.DpDfNet8, nsLevel));
                        break;
                    case "webrtc":
                    case "webrtc-ns":
                        specs.Add(new DnrSpec(DnrKind.WebRtc, "webrtc-ns", DPDFNetModel.DpDfNet8, nsLevel));
                        break;
                    case "dpdfnet":
                        specs.Add(DpdfNet(DPDFNetModel.DpDfNet8));
                        break;
                    case "dpdfnet-all":
                        specs.Add(DpdfNet(DPDFNetModel.Baseline));
                        specs.Add(DpdfNet(DPDFNetModel.DpDfNet2));
                        specs.Add(DpdfNet(DPDFNetModel.DpDfNet4));
                        specs.Add(DpdfNet(DPDFNetModel.DpDfNet8));
                        break;
                    default:
                        if (t.StartsWith("dpdfnet:", StringComparison.Ordinal))
                        {
                            specs.Add(DpdfNet(ParseTier(t.Substring("dpdfnet:".Length))));
                            break;
                        }
                        throw new ArgumentException(
                            $"Unknown backend token: '{raw}' " +
                            "(deepfilternet|dpdfnet[:baseline|2|4|8]|dpdfnet-all|hush|rnnoise|webrtc)");
                }
            }
            return specs;
        }

        private static DnrSpec DpdfNet(DPDFNetModel tier) =>
            new(DnrKind.DpdfNet, "dpdfnet-" + TierLabel(tier), tier, NoiseSuppressionLevel.High);

        private static DPDFNetModel ParseTier(string s) => s switch
        {
            "baseline" => DPDFNetModel.Baseline,
            "2" or "dpdfnet2" => DPDFNetModel.DpDfNet2,
            "4" or "dpdfnet4" => DPDFNetModel.DpDfNet4,
            "8" or "dpdfnet8" => DPDFNetModel.DpDfNet8,
            _ => throw new ArgumentException($"Unknown DPDFNet tier: '{s}' (baseline|2|4|8)"),
        };

        private static string TierLabel(DPDFNetModel tier) => tier switch
        {
            DPDFNetModel.Baseline => "baseline",
            DPDFNetModel.DpDfNet2 => "2",
            DPDFNetModel.DpDfNet4 => "4",
            DPDFNetModel.DpDfNet8 => "8",
            _ => tier.ToString(),
        };
    }
}
