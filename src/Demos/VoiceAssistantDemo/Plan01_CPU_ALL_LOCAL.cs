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
using VoiceBox.LLM.OpenAI;
using VoiceBox.LLM.OnnxGenAI;
using VoiceBox.TTS.MossNano;
using VoiceBox.TTS.Kokoro;
using VoiceBox.TTS.VoxCPM;
using VoiceBox.VAD.FireRed;
using VoiceBox.Wakeup.FsmnKws;
using VoiceBox.Speaker.CampPlus;
using VoiceBox.DNR.DPDFNet.Models;

namespace VoiceAssistantDemo
{
    internal class Plan01_CPU_ALL_LOCAL
    {
        public static OnnxExecutionProvider OnnxExecutionProvider = OnnxExecutionProvider.Cpu;

        public static AssistantRequiredServices Buid(ILoggerFactory? loggerFactory = null)
        {
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
                .AddFireRed2ASRService(options =>
                {
                    options.ModelPath = AudioDefaults.FireRedAsr2Path;
                    options.ExecutionProvider = OnnxExecutionProvider;
                })
                .AddOnnxGenAILLMService(o =>
                {
                    o.ModelPath = AudioDefaults.LlmOnnxModelPath;
                    o.ExecutionProvider = OnnxExecutionProvider.ToGenAIExecutionProvider();
                    o.MaxTokens = 512;
                    o.DoSample = true;
                    o.Temperature = 0.7f;
                    o.TopK = 40;
                    o.TopP = 0.9f;
                    o.RepetitionPenalty = 1.1f;
                })
                .AddVoxCPMTTSService(o =>
                {
                    o.ModelPath = AudioDefaults.VoxCPMTtsPath;
                    o.ExecutionProvider = OnnxExecutionProvider;
                    o.DefaultVoice = AudioDefaults.TtsVoiceNameMap["VoxCPMTTSService"];
                    o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "sichuan_female", PromptWavPath = "./voices/sichuan_female.wav", PromptText = "你晓不晓得，今天天气巴适得很，我准备出去逛一哈，买点菜，晚上回来整火锅。" });
                    o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "sichuan_male", PromptWavPath = "./voices/sichuan_male.wav", PromptText = "你晓不晓得，今天天气巴适得很，我准备出去逛一哈，买点菜，晚上回来整火锅。" });
                    o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "xiaoxiao_female", PromptWavPath = "./voices/xiaoxiao_female.wav", PromptText = "你好，很高兴认识你。今天天气不错，我准备去公园散步，看看花草树木，也听听孩子们的笑声。" });
                    o.CloneVoices.Add(new VoxCpmCloneVoice { Name = "yuanfang_male", PromptWavPath = "./voices/yuanfang_male.wav", PromptText = "你好，很高兴认识你。今天天气不错，我准备去公园散步，看看花草树木，也听听孩子们的笑声。" });
                })
                //.AddMossNanoTTSService(o =>
                //{
                //    o.ModelPath = AudioDefaults.MossNanoTtsPath;
                //    o.ExecutionProvider = OnnxExecutionProvider;
                //    o.DefaultVoice = AudioDefaults.TtsVoiceNameMap["MossNanoTTSService"];
                //    o.ThreadsNum = 4;
                //})
                //.AddKokoroTTSService(o =>
                //{
                //    o.ModelPath = Path.Combine(AudioDefaults.KokoroTtsPath, "kokoro-v1.1-zh.onnx");
                //    o.VoicesPath = Path.Combine(AudioDefaults.KokoroTtsPath, "voices-v1.1-zh.bin");
                //    o.VocabConfigPath = Path.Combine(AudioDefaults.KokoroTtsPath, "config.json");
                //    o.ExecutionProvider = OnnxExecutionProvider;
                //    o.DefaultVoice = AudioDefaults.TtsVoiceNameMap["KokoroTTSService"];
                //    o.ThreadsNum = 4;
                //})
                .Build();

            return AssistantRequiredServices.FromManager(manager);
        }
    }
}
