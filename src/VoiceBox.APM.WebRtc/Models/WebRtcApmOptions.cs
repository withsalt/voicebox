namespace VoiceBox.APM.WebRtc.Models
{
    /// <summary>
    /// <see cref="WebRtcApmService"/> 的配置项。
    /// 该服务是<b>统一采集侧处理器</b>：单个 WebRTC M151 APM 实例在同一处理调用内依次完成
    /// 高通滤波(HPF) → 回声消除(AEC3) → 噪声抑制(NS) → 自动增益(AGC2)。
    /// 启用本服务时，应关闭独立的 HPF/DNR 处理器，避免重复处理。
    /// </summary>
    public sealed class WebRtcApmOptions
    {
        /// <summary>采样率 (Hz)。WebRTC APM 仅支持 8000/16000/32000/48000。默认 16000。</summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>声道数。当前实现仅支持单声道(1)。默认 1。</summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>是否启用高通滤波(去直流/低频隆隆声)。默认 true。</summary>
        public bool EnableHighPassFilter { get; set; } = true;

        /// <summary>是否启用回声消除(AEC3，桌面全功能)。默认 true，是本服务的核心。</summary>
        public bool EnableEchoCanceller { get; set; } = true;

        /// <summary>是否启用噪声抑制。默认 true(统一处理器内一并完成降噪)。</summary>
        public bool EnableNoiseSuppression { get; set; } = true;

        /// <summary>
        /// 是否启用 WebRTC Neural Residual Echo Estimator，在 AEC3 线性消除后用 ML 模型估计残余回声。
        /// 默认 false。启用后必须同时启用 <see cref="EnableEchoCanceller"/>。
        /// </summary>
        public bool EnableNeuralResidualEchoEstimator { get; set; }

        /// <summary>
        /// Neural Residual Echo Estimator 的 TFLite 模型路径。默认 null，使用包内
        /// <c>runtimes/models/aec/webrtc/v20260212/model.tflite</c>；相对路径以应用输出目录为基准。
        /// 可用于验证新模型或在不升级程序集的情况下切换模型版本。
        /// </summary>
        public string? NeuralResidualEchoEstimatorModelPath { get; set; }

        /// <summary>
        /// 噪声抑制强度（仅 <see cref="EnableNoiseSuppression"/> 为 true 时生效）。
        /// 默认 <see cref="Models.NoiseSuppressionLevel.High"/>，在抑制强度与语音自然度间取平衡。
        /// </summary>
        public NoiseSuppressionLevel NoiseSuppressionLevel { get; set; } = NoiseSuppressionLevel.High;

        /// <summary>
        /// 是否启用 AGC2（现代自适应数字增益 + 限幅），把近端语音归一到合适电平、利于下游 ASR。
        /// <para>
        /// <b>默认 false。</b>AGC2 的自适应数字增益在安静/低电平时会把噪声底与 AEC 残差猛拉增益，
        /// 在"麦克风→扬声器"回环里更会形成正反馈 → 断续嗡嗡声/啸叫。旧的可用 AEC 实现亦全程关闭 AGC。
        /// 仅在确有电平归一需求、且能接受上述风险时显式开启；建议先在 native shim 暴露
        /// AGC2 降噪门限/最大增益等安全参数后再启用。
        /// </para>
        /// </summary>
        public bool EnableAgc2 { get; set; } = false;

        /// <summary>是否启用 AGC2 自适应数字增益级。默认 false(仅 <see cref="EnableAgc2"/> 为 true 时生效)。</summary>
        public bool Agc2AdaptiveDigital { get; set; } = false;

        /// <summary>
        /// AEC3 双讲检测 <c>dominant_nearend_detection.enr_threshold</c>。
        /// 控制何时判定"近端主导"(双讲)，越低越倾向保留近端语音。WebRTC 默认 0.25。
        /// </summary>
        public float DominantNearendEnrThreshold { get; set; } = 0.25f;

        /// <summary>AEC3 <c>dominant_nearend_detection.enr_exit_threshold</c>。退出近端主导态的阈值。默认 10。</summary>
        public float DominantNearendEnrExitThreshold { get; set; } = 10.0f;

        /// <summary>AEC3 <c>dominant_nearend_detection.snr_threshold</c>。近端主导判定的信噪比阈值。默认 30。</summary>
        public float DominantNearendSnrThreshold { get; set; } = 30.0f;

        /// <summary>AEC3 <c>ep_strength.nearend_len</c>。近端回声路径长度估计权重，影响双讲下抑制强度。默认 0.83。</summary>
        public float EchoPathNearendLen { get; set; } = 0.83f;

        /// <summary>校验配置。无效时抛 <see cref="ArgumentException"/>。</summary>
        public void Validate()
        {
            if (SampleRate is not (8000 or 16000 or 32000 or 48000))
                throw new ArgumentException(
                    $"WebRtcApmOptions.SampleRate must be 8000/16000/32000/48000, got {SampleRate}.");
            if (ChannelCount != 1)
                throw new ArgumentException(
                    $"WebRtcApmService currently supports mono only (ChannelCount=1), got {ChannelCount}.");
            if (EnableNeuralResidualEchoEstimator && !EnableEchoCanceller)
                throw new ArgumentException(
                    "EnableNeuralResidualEchoEstimator requires EnableEchoCanceller=true.");
            if (EnableNeuralResidualEchoEstimator &&
                NeuralResidualEchoEstimatorModelPath is not null &&
                string.IsNullOrWhiteSpace(NeuralResidualEchoEstimatorModelPath))
            {
                throw new ArgumentException(
                    "NeuralResidualEchoEstimatorModelPath cannot be empty when the neural estimator is enabled.");
            }
        }
    }
}
