using VoiceBox.Core.Onnx;

namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>
    /// Configuration for <see cref="DfsmnAecService"/>.
    /// The model is trained for 16 kHz mono audio; other formats should be converted before reaching this service.
    /// </summary>
    public class DfsmnAecOptions : OnnxExecutionOptions
    {
        public const int DefaultFeatureFrameLength = 640; // 40 ms
        public const int DefaultHopLength = 320;          // 20 ms
        public const int DefaultMaskBins = 321;           // n_fft / 2 + 1
        public const int DefaultFeatureBins = 80;
        public const int DefaultFeatureDim = 240;

        private const uint SupportedSampleRate = 16000;
        private string? _modelPath;
        private string? _mvnPath;

        /// <summary>
        /// ONNX Runtime intra-op 线程数。DFSMN AEC 的 ONNX 模型较小，且运行在短时流式 hop 上，过多线程反而可能增加调度开销和回调竞争。
        /// </summary>
        public override int ThreadsNum { get; set; } = 1;

        /// <summary>Input sample rate. Only 16 kHz is supported by this model package.</summary>
        public uint SampleRate { get; set; } = SupportedSampleRate;

        /// <summary>Input channel count. Only mono is supported.</summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>
        /// Streaming DFSMN core ONNX model path. When null, the built-in mask+VAD runtime model path is used.
        /// The ONNX core consumes normalized fbank features and returns spectral masks plus the auxiliary
        /// near-end VAD probability output.
        /// </summary>
        public string? ModelPath
        {
            get => _modelPath;
            set => _modelPath = string.IsNullOrWhiteSpace(value) ? null : value;
        }

        /// <summary>Kaldi-style MVN transform path (feature.DEY.mvn.txt).</summary>
        public string? MvnPath
        {
            get => _mvnPath;
            set => _mvnPath = string.IsNullOrWhiteSpace(value) ? null : value;
        }

        /// <summary>
        /// Delay the render reference by this many samples before feeding the built-in linear AEC.
        /// Use this when the speaker-to-microphone echo arrives later than the full-duplex callback's render span.
        /// At 16 kHz, 160 samples = 10 ms.
        /// </summary>
        public int ReferenceDelaySamples { get; set; }

        /// <summary>Maximum absolute audio sample after processing.</summary>
        public float ClipLimit { get; set; } = 1.0f;

        /// <summary>
        /// Optional attenuation floor in dB for neural masking. For example, 48 limits masks to at least -48 dB.
        /// Null allows the model mask unchanged.
        /// </summary>
        public float? MaskFloorDb { get; set; } = 60f;

        /// <summary>Built-in linear echo canceller settings.</summary>
        public DfsmnAecLinearOptions Linear { get; set; } = new DfsmnAecLinearOptions();

        /// <summary>Residual echo suppressor settings.</summary>
        public DfsmnAecResidualOptions Residual { get; set; } = new DfsmnAecResidualOptions();

        /// <summary>
        /// Double-talk detector settings. Freezes linear-AEC adaptation while the near-end talker
        /// speaks over the echo, so a barge-in is not cancelled along with the far-end audio.
        /// </summary>
        public DfsmnAecDoubleTalkOptions DoubleTalk { get; set; } = new DfsmnAecDoubleTalkOptions();

        /// <summary>
        /// Model-gated near-end protection: uses the per-hop near-end VAD probability (emitted by
        /// the same inference as the mask) to bound mask/residual suppression during double-talk,
        /// so a barging-in talker survives to downstream VAD/ASR. Ignored by mask-only models.
        /// </summary>
        public DfsmnAecNearEndProtectionOptions NearEndProtection { get; set; } = new DfsmnAecNearEndProtectionOptions();

        internal void Validate()
        {
            if (SampleRate != SupportedSampleRate)
                throw new NotSupportedException($"DFSMN AEC supports {SupportedSampleRate} Hz only, got {SampleRate}.");
            if (ChannelCount != 1)
                throw new NotSupportedException($"DFSMN AEC supports mono audio only, got {ChannelCount} channels.");
            if (Linear is null)
                throw new ArgumentNullException(nameof(Linear), "Linear options cannot be null.");
            Linear.Validate();
            if (ReferenceDelaySamples < 0)
                throw new ArgumentOutOfRangeException(nameof(ReferenceDelaySamples), ReferenceDelaySamples, "ReferenceDelaySamples must be >= 0.");
            if (ClipLimit <= 0 || ClipLimit > 1 || float.IsNaN(ClipLimit))
                throw new ArgumentOutOfRangeException(nameof(ClipLimit), ClipLimit, "ClipLimit must be in (0, 1].");
            if (MaskFloorDb is { } floor && (floor < 0 || float.IsNaN(floor)))
                throw new ArgumentOutOfRangeException(nameof(MaskFloorDb), floor, "MaskFloorDb must be non-negative or null.");
            if (Residual is null)
                throw new ArgumentNullException(nameof(Residual), "Residual options cannot be null.");
            Residual.Validate();
            if (DoubleTalk is null)
                throw new ArgumentNullException(nameof(DoubleTalk), "DoubleTalk options cannot be null.");
            DoubleTalk.Validate();
            if (NearEndProtection is null)
                throw new ArgumentNullException(nameof(NearEndProtection), "NearEndProtection options cannot be null.");
            NearEndProtection.Validate();
        }
    }
}
