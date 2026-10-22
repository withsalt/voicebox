using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.APM.WebRtc;
using VoiceBox.ASR.DoubaoOnline;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.MiniAudio;
using VoiceBox.Core;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.DNR.DPDFNet;
using VoiceBox.LLM.OpenAI;
using VoiceBox.VAD.FireRed;
using VoiceBox.Wakeup.FsmnKws;
using VoiceBox.Speaker.CampPlus;
using VoiceBox.TTS.DoubaoOnline;
using VoiceBox.TTS.DoubaoOnline.SpeakerCatalog;
using VoiceBox.DNR.DPDFNet.Models;

namespace VoiceAssistantDemo
{
    internal class Plan02_CPU_ALL_ONLINE
    {
        public static AssistantRequiredServices Buid(ILoggerFactory? loggerFactory = null)
        {
            string asrApiKey = GetRequiredEnvironmentVariable("VOLC_ASR_API_KEY");
            string llmApiKey = GetRequiredEnvironmentVariable("OPENAI_API_KEY");
            string ttsApiKey = GetRequiredEnvironmentVariable("VOLC_TTS_API_KEY");


            IAudioServiceManager manager = new AudioServiceManager()
                .AddLogger(loggerFactory ?? NullLoggerFactory.Instance)
                //.AddPortAudioDuplexService(o =>
                //{
                //    o.SampleRate = AudioDefaults.SampleRate;
                //    o.ChannelCount = AudioDefaults.ChannelCount;
                //    o.SampleFormat = SampleFormat.Float32;
                //    o.SuggestedLatency = 0.020;
                //    o.MaxFrameSize = 256;
                //    o.RecordBufferCapacity = 10;
                //    o.PlayMaxQueueSize = 60;
                //    o.EnableWasapiExclusiveMode = false;
                //    o.EnableAec = true;             // 回声消除由 WebRTC APM 处理
                //    o.EnableDnr = AudioDefaults.UsesIndependentDnr;
                //    o.EnableHpf = false;            // 高通滤波已由 WebRTC APM 处理
                //})
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
                    o.EnableNoiseSuppression = AudioDefaults.UsesWebRtcNoiseSuppression;
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
                })
                .AddFireRedVoiceActivityDetectService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.SpeechThreshold = 0.9f;
                })
                .AddFsmnKwsService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.ModelPath = Path.Combine(AudioDefaults.WakeupKwsPath, "kws_streaming.onnx");
                    o.Threshold = 0.5f;
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
                .AddDoubaoOnlineASRService(options =>
                {
                    options.ApiKey = asrApiKey;
                    options.StreamMode = SeedAsrStreamMode.Bidirectional;
                    options.PackageDurationMs = 100;
                })
                .AddOpenAILLMService(o =>
                {
                    o.ApiKey = llmApiKey;
                    o.Endpoint = AudioDefaults.LlmEndpoint;
                    o.Model = AudioDefaults.LlmModelName;
                    o.MaxTokens = 4096;
                    o.Thinking = AudioDefaults.IsEnableThinking ? ThinkingMode.Enabled : ThinkingMode.Disabled;
                })
                .AddDoubaoOnlineTTSService(o =>
                {
                    o.ApiKey = ttsApiKey;
                    o.DefaultVoice = AudioDefaults.TtsVoiceNameMap["DoubaoOnlineTTSService"];
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.SessionTimeoutSec = 10;
                })
                .Build();

            return AssistantRequiredServices.FromManager(manager);
        }

        private static string GetRequiredEnvironmentVariable(string name)
            => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value
                ? value
                : throw new InvalidOperationException($"Missing required environment variable: {name}.");
    }
}
