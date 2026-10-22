using System.Runtime.InteropServices;

namespace VoiceBox.APM.WebRtc.Internal
{
    /// <summary>
    /// P/Invoke 绑定，对应原生 <c>webrtc-apm</c>(C ABI，见 <c>native/webrtc-apm/include/apm_shim.h</c>)。
    /// 库名用裸名 <c>webrtc-apm</c>，运行时按平台自动解析为
    /// <c>webrtc-apm.dll</c> / <c>libwebrtc-apm.so</c> / <c>libwebrtc-apm.dylib</c>。
    /// </summary>
    internal static partial class ApmInterop
    {
        private const string Lib = "webrtc-apm";

        /// <summary>本绑定所期望的原生 ABI 版本，须与 <c>APM_ABI_VERSION</c> 一致。</summary>
        public const int ExpectedAbiVersion = 4;

        // 状态码(与 apm_shim.h 对应)。0 成功；[-99,-1] 透传 WebRTC 错误；&lt;=-100 为 shim 自有。
        public const int Ok = 0;
        public const int ErrNullArg = -100;
        public const int ErrAbi = -101;
        public const int ErrInternal = -102;
        public const int ErrCreate = -103;
        public const int ErrBadArg = -104;
        public const int ErrModel = -105;

        /// <summary>创建配置，布局须与原生 <c>ApmConfig</c> 严格一致(顺序/类型/对齐)。</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ApmConfig
        {
            public uint StructSize;

            public int SampleRate;
            public int NumChannels;

            public int EnableHighPassFilter;
            public int EnableEchoCanceller;
            public int EnableNoiseSuppression;
            public int NsLevel;
            public int EnableAgc2;
            public int Agc2AdaptiveDigital;

            public int StreamDelayMs;

            public float DtEnrThreshold;
            public float DtEnrExitThreshold;
            public float DtSnrThreshold;
            public float EpNearendLen;
        }

        /// <summary>统计快照，布局须与原生 <c>ApmStats</c> 严格一致。每项以 has/value 成对表达可选值。</summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct ApmStats
        {
            public int HasEchoReturnLoss;
            public double EchoReturnLoss;

            public int HasEchoReturnLossEnhancement;
            public double EchoReturnLossEnhancement;

            public int HasDelayMs;
            public int DelayMs;

            public int HasDelayMedianMs;
            public int DelayMedianMs;

            public int HasResidualEchoLikelihood;
            public double ResidualEchoLikelihood;

            public int HasResidualEchoLikelihoodRecentMax;
            public double ResidualEchoLikelihoodRecentMax;

            public int HasDivergentFilterFraction;
            public double DivergentFilterFraction;
        }

        [LibraryImport(Lib, EntryPoint = "apm_abi_version")]
        public static partial int AbiVersion();

        [LibraryImport(Lib, EntryPoint = "apm_create")]
        public static partial int Create(in ApmConfig config, out nint handle);

        [LibraryImport(Lib, EntryPoint = "apm_create_with_neural_residual_echo_model")]
        public static unsafe partial int CreateWithNeuralResidualEchoModel(
            in ApmConfig config,
            byte* modelData,
            ulong modelSize,
            out nint handle);

        [LibraryImport(Lib, EntryPoint = "apm_destroy")]
        public static partial void Destroy(nint handle);

        [LibraryImport(Lib, EntryPoint = "apm_set_stream_delay_ms")]
        public static partial int SetStreamDelayMs(nint handle, int delayMs);

        [LibraryImport(Lib, EntryPoint = "apm_process_reverse")]
        public static unsafe partial int ProcessReverse(nint handle, float** channels, int numChannels, int numFrames);

        [LibraryImport(Lib, EntryPoint = "apm_process_capture")]
        public static unsafe partial int ProcessCapture(nint handle, float** channels, int numChannels, int numFrames);

        [LibraryImport(Lib, EntryPoint = "apm_get_stats")]
        public static partial int GetStats(nint handle, out ApmStats stats);

        [LibraryImport(Lib, EntryPoint = "apm_reset")]
        public static partial int Reset(nint handle);

        // ---- 独立 AEC3 延迟估计器 (ABI v3, 见 apm_shim.h 的 apm_delay_*) ----

        [LibraryImport(Lib, EntryPoint = "apm_delay_create")]
        public static partial int DelayCreate(int sampleRate, int numChannels, out nint handle);

        [LibraryImport(Lib, EntryPoint = "apm_delay_destroy")]
        public static partial void DelayDestroy(nint handle);

        /// <summary>返回 1=有新估计(写出参)，0=无新估计，&lt;0=错误状态码。</summary>
        [LibraryImport(Lib, EntryPoint = "apm_delay_process")]
        public static unsafe partial int DelayProcess(nint handle, float* render, float* capture,
            int numSamples, int* outDelaySamples, int* outQuality);

        [LibraryImport(Lib, EntryPoint = "apm_delay_reset")]
        public static partial int DelayReset(nint handle);
    }
}
