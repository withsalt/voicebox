using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.APM.WebRtc;
using VoiceBox.ASR.Qwen3;
using VoiceBox.Audio.PortAudio;
using VoiceBox.Audio.MiniAudio;
using VoiceBox.Core;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.LLM.LLamaSharp;
using VoiceBox.TTS.Qwen3;
using VoiceBox.VAD.FireRed;
using VoiceBox.Wakeup.FsmnKws;
using VoiceBox.Speaker.CampPlus;
using VoiceBox.Speaker.WeSpeaker;
using VoiceBox.DNR.DPDFNet;
using VoiceBox.DNR.DPDFNet.Models;
using VoiceBox.HPF.Butterworth;
using VoiceBox.ASR.Confucius4;
using VoiceBox.TTS.VoxCPM;
using System.Runtime.InteropServices;

namespace VoiceAssistantDemo
{
    internal class Plan03_GPU_16GVRAM_ALL_LOCAL
    {
        public static OnnxExecutionProvider OnnxExecutionProvider = OnnxExecutionProvider.Cuda;

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
                //    o.EnableDnr = false;
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
                    o.EnableNoiseSuppression = true;
                    o.EnableNeuralResidualEchoEstimator = true;
                    o.EnableAgc2 = false;
                    // 调低会更容易保留用户插话，也可能多留一些回声；WebRTC 默认值为 0.25。
                    //o.DominantNearendEnrThreshold = 0.35f;
                })
                //.AddDPDFNetDNRService(o =>
                //{
                //    o.SampleRate = AudioDefaults.SampleRate;
                //    o.ChannelCount = AudioDefaults.ChannelCount;
                //    o.Model = DPDFNetModel.DpDfNet2;
                //    o.AttenuationLimitDb = 12f;
                //    o.AttenuationMode = DPDFNetAttenuationMode.MagnitudeFloor;
                //})
                // ── 切回原方案：注释掉上面的 AddWebRtcApmService、取消下面三段注释即可 ───────────
                //.AddDfsmnAecService(o =>
                //{
                //    o.SampleRate = AudioSampleRate;
                //    o.ChannelCount = AudioChannelCount;
                //    // 双讲 (barge-in) 检测: 用户在 TTS 播放期开口时, 冻结线性 AEC 的自适应, 防止把近端
                //    // 语音当成滤波器误差去抵消 —— 否则滤波器/神经掩码会被带偏, 连用户的声音一起抹掉,
                //    // VAD 就检测不到、打断失效。默认已开启; 下列为可凭听感微调的灵敏度旋钮:
                //    o.DoubleTalk.NearEndThreshold = 2.0f;  // 调小=更早冻结(更护近端, 回声可能多漏); 调大=更保守
                //    o.DoubleTalk.HangoverMs = 120f;        // 词间停顿时维持冻结的时长
                //})
                //.AddDPDFNetDNRService(o =>
                //{
                //    o.SampleRate = AudioSampleRate;
                //    o.ChannelCount = AudioChannelCount;
                //    o.Model = DPDFNetModel.DpDfNet2;
                //    o.AttenuationLimitDb = 12f;
                //})
                //.AddButterworthHighPassFilterService(o =>
                //{
                //    o.SampleRate = AudioSampleRate;
                //    o.ChannelCount = AudioChannelCount;
                //    o.CutoffFrequency = 100;  // 截止频率 100Hz，滤除直流偏置与低频隆隆声
                //    o.Order = 2;              // 二阶 Butterworth
                //})
                //.AddRNNoiseDNRService(options =>
                //{
                //    options.SampleRate = 16000;
                //    options.ChannelCount = 1;
                //    options.SampleFormat = SampleFormat.Float32;
                //})
                .AddFireRedVoiceActivityDetectService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.SampleFormat = SampleFormat.Float32;
                    o.ThreadsNum = 1;
                    o.SpeechThreshold = 0.9f;
                })
                .AddFsmnKwsService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = AudioDefaults.ChannelCount;
                    o.ModelPath = Path.Combine(AudioDefaults.WakeupKwsPath, "kws_streaming.onnx");
                    o.Threshold = 0.25f;
                })
                //.AddCampPlusSpeakerService(o =>
                //{
                //    o.SampleRate = AudioDefaults.SampleRate;
                //    o.ChannelCount = AudioDefaults.ChannelCount;
                //    o.SampleFormat = SampleFormat.Float32;
                //    o.WindowMs = 500;
                //    o.HopMs = 100;
                //    o.ScoreSmoothing = 1.0f;
                //})
                .AddWeSpeakerService(o =>
                {
                    o.SampleRate = AudioDefaults.SampleRate;
                    o.ChannelCount = 1;
                    o.SampleFormat = SampleFormat.Float32;
                    o.WindowMs = 500;   // 与非 LM 模型的 200 帧训练长度对齐
                    o.HopMs = 200;
                    o.ScoreSmoothing = 0.6f;
                    o.Threshold = 0.3f;
                    o.MinWindowRms = 0f;       // 不跳过静音窗，持续产出每个滑动窗口的结果
                })
                .AddConfucius4ASRService(options =>
                {
                    options.ModelPath = AudioDefaults.Confucius4AsrPath;
                    options.ExecutionProvider = OnnxExecutionProvider;
                })
                .AddLLamaSharpLLMService(o =>
                {
                    o.ModelPath = AudioDefaults.LlmGgufModelPath;
                    o.ContextSize = 4096;
                    o.GpuLayerCount = -1;        // 纯 CPU; 有 GPU 时改为 -1 (全部卸载)
                    o.MaxTokens = 4096;
                    // 以下采样参数适合对话场景
                    o.DoSample = true;
                    o.Temperature = 0.7f;
                    o.TopK = 40;
                    o.TopP = 0.9f;
                    o.RepetitionPenalty = 1.1f;
                    o.EnableSessionCache = true;
                })
                //.AddQwen3TTSService(o =>
                //{
                //    o.ModelPath = AudioDefaults.Qwen3TtsPath;
                //    o.ExecutionProvider = OnnxExecutionProvider;
                //    o.DefaultVoice = AudioDefaults.TtsVoiceNameMap["Qwen3TTSService"]; ;
                //    o.StreamChunkSchedule = [1, 2, 2, 4, 4, 8];
                //})
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
                .Build();

            return AssistantRequiredServices.FromManager(manager);
        }

        private static string GetRequiredEnvironmentVariable(string name)
            => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value
                ? value
                : throw new InvalidOperationException($"Missing required environment variable: {name}.");
    }
}
