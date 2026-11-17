using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.APM.WebRtc;
using VoiceBox.ASR.FireRedASR2;
using VoiceBox.ASR.Qwen3;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.MiniAudio;
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DNR.DPDFNet;
using VoiceBox.LLM.LLamaSharp;
using VoiceBox.TTS.Hojo;
using VoiceBox.VAD.FireRed;
using VoiceBox.Wakeup.FsmnKws;
using VoiceBox.Speaker.CampPlus;
using VoiceBox.DNR.DPDFNet.Models;

namespace VoiceAssistantDemo
{
    internal class Plan01_CPU_8GRAM_ALL_LOCAL
    {
        public static OnnxExecutionProvider OnnxExecutionProvider = OnnxExecutionProvider.Cpu;

        public static AssistantRequiredServices Buid(ILoggerFactory? loggerFactory = null)
        {
            IAudioServiceManager manager = new AudioServiceManager()
                .AddLogger(loggerFactory ?? NullLoggerFactory.Instance)
                .AddMiniAudioDuplexService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.SuggestedLatency = 0.020;
                    o.MaxFrameSize = 256;
                    o.RecordBufferCapacity = 10;
                    o.PlayMaxQueueSize = 60;
                    o.EnableWasapiExclusiveMode = false;
                    o.EnableAec = true;             // 回声消除由 WebRTC APM 处理
                    o.EnableDnr = false;
                    o.EnableHpf = false;            // 高通滤波已由 WebRTC APM 处理
                })
                // APM 负责高通、回声消除和可选的噪声抑制。
                // 关闭 AGC，避免安静时把环境底噪一起放大。
                .AddWebRtcApmService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.EnableHighPassFilter = true;
                    o.EnableEchoCanceller = true;
                    o.EnableNoiseSuppression = true;
                    o.EnableAgc2 = false;
                    // 调低会更容易保留用户插话，也可能多留一些回声；WebRTC 默认值为 0.25。
                    //o.DominantNearendEnrThreshold = 0.35f;
                })
                .AddDPDFNetDNRService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.Model = DPDFNetModel.DpDfNet2;
                    o.AttenuationLimitDb = 12f;
                    o.AttenuationMode = DPDFNetAttenuationMode.MagnitudeFloor;
                    o.ExecutionProvider = OnnxExecutionProvider;
                })
                .AddFireRedVoiceActivityDetectService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.SpeechThreshold = 0.6f;
                })
                .AddFsmnKwsService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.ModelPath = Path.Combine(AudioDefaults.WakeupKwsPath, "kws_streaming.onnx");
                    o.Threshold = 0.5f;
                    o.DebounceMs = 100;
                })
                .AddCampPlusSpeakerService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.WindowMs = 500;
                    o.HopMs = 100;
                    o.ScoreSmoothing = 1.0f;
                })
                .AddFireRed2ASRService(o =>
                {
                    o.ModelPath = AudioDefaults.FireRedAsr2Path;
                    o.ExecutionProvider = OnnxExecutionProvider;
                })
                .AddHojoTTSService(o =>
                {
                    o.ModelPath = AudioDefaults.HojoTtsPath;
                    o.DefaultVoice = AudioDefaults.TtsVoiceNameMap["HojoTTSService"];
                    o.ThreadsNum = 2;
                    o.ChunkFrames = 40;
                    o.ExecutionProvider = OnnxExecutionProvider;
                })
                .AddLLamaSharpLLMService(o =>
                {
                    o.ModelPath = AudioDefaults.LlmGgufModelPath;
                    o.ContextSize = 8192;
                    o.GpuLayerCount = 0;
                    o.MaxTokens = 512;
                    o.DoSample = true;
                    o.Temperature = 0.7f;
                    o.TopK = 40;
                    o.TopP = 0.9f;
                    o.RepetitionPenalty = 1.1f;
                })
                .Build();

            return AssistantRequiredServices.FromManager(manager);
        }
    }
}
