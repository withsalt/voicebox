using System;
using System.Collections.Generic;
using System.Text;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AECService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.DNRService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Services.WakeupService;
using VoiceBox.Core.Services.HPFService;

namespace VoiceBox.Core
{
    public interface IAudioServiceManager : IDisposable
    {
        /// <summary>
        /// 录音
        /// </summary>
        /// <returns></returns>
        IAudioRecordService GetAudioRecordInstance(string? name = null);

        /// <summary>
        /// 音频播放
        /// </summary>
        /// <returns></returns>
        IAudioPlayService GetAudioPlayInstance(string? name = null);

        /// <summary>
        /// 全双工录放音频
        /// </summary>
        /// <returns></returns>
        IAudioDuplexService GetAudioDuplexInstance(string? name = null);

        /// <summary>
        /// VAD
        /// </summary>
        /// <returns></returns>
        IVoiceActivityDetectService GetVoiceActivityDetectInstance(string? name = null);

        /// <summary>
        /// 实时降噪
        /// </summary>
        /// <returns></returns>
        IDigitalNoiseReductionService GetDNRServiceInstance(string? name = null);

        /// <summary>
        /// 声学回声消除
        /// </summary>
        /// <returns></returns>
        IAcousticEchoCancellationService GetAECServiceInstance(string? name = null);

        /// <summary>
        /// 获取可选的声学处理指标源。
        /// </summary>
        IAcousticProcessingMetricsSource GetAcousticProcessingMetricsSource(string? name = null);

        /// <summary>
        /// 高通滤波
        /// </summary>
        /// <returns></returns>
        IHighPassFilterService GetHighPassFilterServiceInstance(string? name = null);

        /// <summary>
        /// 流式（实时）语音识别
        /// </summary>
        /// <returns></returns>
        IStreamASRService GetStreamASRServiceInstance(string? name = null);

        /// <summary>
        /// 非实时语音识别
        /// </summary>
        /// <returns></returns>
        IASRService GetOfflineASRServiceInstance(string? name = null);

        /// <summary>
        /// 语音唤醒
        /// </summary>
        /// <returns></returns>
        IWakeupService GetWakeupServiceInstance(string? name = null);

        /// <summary>
        /// 说话人验证 (声纹)
        /// </summary>
        /// <returns></returns>
        ISpeakerService GetSpeakerServiceInstance(string? name = null);

        /// <summary>
        /// 语音合成 (TTS)
        /// </summary>
        /// <returns></returns>
        ITTSService GetTTSServiceInstance(string? name = null);

        /// <summary>
        /// 流式语音合成 (TTS)
        /// </summary>
        /// <returns></returns>
        IStreamTTSService GetStreamTTSServiceInstance(string? name = null);

        /// <summary>
        /// 大语言模型推理 (LLM)
        /// </summary>
        /// <returns></returns>
        ILLMService GetLLMServiceInstance(string? name = null);

        /// <summary>
        /// 流式大语言模型推理 (LLM)
        /// </summary>
        /// <returns></returns>
        IStreamLLMService GetStreamLLMServiceInstance(string? name = null);
    }
}
